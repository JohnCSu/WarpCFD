import warp as wp
from warp_cfd.FV import FVM
import numpy as np
from warp_cfd.FV.terms import ConvectionTerm,DiffusionTerm, GradTerm,Matrix
from warp_cfd.FV.field import Field
import warp as wp
from warp_cfd.FV.Ops.array_ops import sub_1D_array,add_1D_array,div_1D_array
from warp_cfd.FV.Ops.fv_ops import interpolate_cell_value_to_face,calculate_rUA,get_HbyA,divFlux
from warp.types import vector
from warp.optim import linear
class SIMPLE():
    def __init__(self,model:FVM,u_relaxation_factor=0.7,p_relaxation_factor = 0.3,correction = False) -> None:
        self.model = model

        self.float_dtype = model.float_dtype
        velocity_vars = ['u','v','w']
        self.convection = ConvectionTerm(model,velocity_vars,'upwind') # We only want velocities
        self.diffusion = DiffusionTerm(model,velocity_vars,correction = correction)
        self.grad_P = GradTerm(model,'p') # P

        self.vel_equation = Matrix(model,fields = velocity_vars)

        self.p_correction_diffusion = DiffusionTerm(model,'p',need_global_index= True,von_neumann= 0.,correction=correction)
        self.p_corr_equation = Matrix(model,fields = 'p',solver = linear.cg)

        self.vel_correction = wp.zeros(shape=(model.num_cells,3),dtype=float)

        self.vel_array = wp.zeros(shape=(model.num_cells*3),dtype=model.float_dtype)

        self.p_relaxation_factor = p_relaxation_factor
        self.u_relaxation_factor = u_relaxation_factor
        self.NUM_INNER_LOOPS = 1

        self.HbyA = wp.zeros_like(self.vel_array)
        self.grad_P_HbyA = wp.zeros_like(self.vel_array)
        self.rUA = wp.zeros(shape= model.cells.shape[0],dtype= model.float_dtype)
        self.rUA_faces = wp.zeros(shape= model.faces.shape[0],dtype= model.float_dtype)
        self.HbyA_faces = wp.zeros(shape=model.faces.shape[0],dtype = vector(3,dtype=model.float_dtype))
    def run(self,num_steps,steps_per_check = 10,*,rhie_chow = True):
        model = self.model
        convection = self.convection
        diffusion = self.diffusion
        grad_P = self.grad_P
        p_correction_diffusion = self.p_correction_diffusion
        vel_equation= self.vel_equation
        p_corr_equation = self.p_corr_equation
        vel_array = self.vel_array
        HbyA = self.HbyA
        rUA = self.rUA 
        rUA_faces = self.rUA_faces
        for i in range(num_steps):
            model.set_boundary_conditions()
            model.face_interpolation()
            model.calculate_gradients()
            model.calculate_mass_flux()

            # intermediate Velocity Step
            convection(model)
            diffusion(model,viscosity = model.viscosity)
            grad_P(model)
            vel_equation.form_system([convection,-diffusion],explicit_terms= -grad_P,fvm = model)
            # print('u\n',vel_equation.dense[::3,::3])
            vel_equation.relax(self.u_relaxation_factor,model)
            # 
            # print('v\n',vel_equation.dense[1::3,1::3])
            Ap = vel_equation.diagonal
            outer_loop_result,vel_array = vel_equation.solve_Axb(vel_array)
            # print(vel_array[::3])
     
            rUA = calculate_rUA(Ap,model.cells,rUA)
            rUA_faces = interpolate_cell_value_to_face(rUA_faces,rUA,model.faces)

            p_grad = model.get_gradient('p').flatten() 
            HbyA = get_HbyA(HbyA,rUA,vel_array,p_grad)
            model.replace_cell_values(['u','v','w'],HbyA)
            
            for _ in range(self.NUM_INNER_LOOPS):
                
                model.face_interpolation()
                model.calculate_gradients() # We need P grad
                model.calculate_mass_flux()

                # Pressure Correction
                div_u = model.divFlux()
                
                p_correction_diffusion(model,viscosity = rUA_faces)

                p_corr_equation.form_system(p_correction_diffusion,fvm = model)
                p_corr_equation.add_RHS(div_u)
                p_corr_equation.replace_row(0,0.)
                
                inner_loop_result,p_cor = p_corr_equation.solve_Axb()

                model.relax(p_cor,alpha = self.p_relaxation_factor, output_index= 3)
                
                model.face_interpolation('p')
                model.calculate_gradients('p')
                vel_correction = model.get_gradient('p',coeff = rUA).flatten()
                
                sub_1D_array(HbyA,vel_correction,HbyA)
                model.replace_cell_values(['u','v','w'],HbyA)
                

            if i % steps_per_check == 0:
                print(f'step iter: {i}')
                print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
                converged = model.check_convergence(vel_equation.matrix,vel_array,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
                
                if converged:
                    print(f'Run Reached Convergence Criteria at iteration {i}')
                    return None
                
        print(f'MAX ITERATIONS OF {num_steps} REACHED. Terminating')
        print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
        converged = model.check_convergence(vel_equation.matrix,vel_array,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
