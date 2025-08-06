import warp as wp
from warp_cfd.FV import FVM
import numpy as np
from warp_cfd.FV.terms import ConvectionTerm,DiffusionTerm, GradTerm,Equation
from warp_cfd.FV.field import Field
import warp as wp
from warp_cfd.FV.kernels.solver_kernels import interpolate_cell_value_to_face,calculate_rUA,get_HbyA
from warp.types import vector
from warp.optim import linear
class IncompressibleSolver():
    def __init__(self,model:FVM,u_relaxation_factor=0.7,p_relaxation_factor = 0.3,correction = False) -> None:
        self.model = model

        self.float_dtype = model.float_dtype
        velocity_vars = ['u','v','w']
        self.convection = ConvectionTerm(model,velocity_vars,'upwind') # We only want velocities
        self.diffusion = DiffusionTerm(model,velocity_vars,correction = correction)
        self.grad_P = GradTerm(model,'p') # P

        self.vel_equation = Equation(model,fields = velocity_vars)

        self.p_correction_diffusion = DiffusionTerm(model,'p',correction=correction)
        self.p_corr_equation = Equation(model,fields = 'p',solver = linear.cg)

        self.vel_correction = wp.zeros(shape=(model.num_cells,3),dtype=float)

        
        self.p_relaxation_factor = p_relaxation_factor
        self.u_relaxation_factor = u_relaxation_factor
        self.NUM_INNER_LOOPS = 1

        self.material = model.material
        
        self.HbyA = wp.zeros(shape=(model.num_cells*3),dtype=model.float_dtype)
        # self.grad_P_HbyA = wp.zeros_like(self.vel_array)
        self.rUA = wp.zeros(shape= model.cells.shape[0],dtype= model.float_dtype)
        self.rUA_faces = wp.zeros(shape= model.faces.shape[0],dtype= model.float_dtype)
        self.HbyA_faces = wp.zeros(shape=model.faces.shape[0],dtype = vector(3,dtype=model.float_dtype))
    def run(self,num_steps,steps_per_check = 10):
        model = self.model
        convection = self.convection
        diffusion = self.diffusion
        grad_P = self.grad_P
        p_correction_diffusion = self.p_correction_diffusion
        vel_equation= self.vel_equation
        p_corr_equation = self.p_corr_equation
        
        HbyA = self.HbyA
        rUA = self.rUA 
        rUA_faces = self.rUA_faces


        inv_density = (1./self.material.density)
        density = self.material.density
        kinematic_viscosity = self.material.viscosity/self.material.density

        for i in range(num_steps):
            model.set_boundary_conditions()
            model.face_interpolation()
            model.calculate_gradients()
            model.calculate_mass_flux()

            # intermediate Velocity Step
            convection(model)
            diffusion(model,viscosity = kinematic_viscosity)
            grad_P(model)
            vel_equation.form_system([convection,-diffusion],explicit_terms= -inv_density*grad_P,fvm = model)
            
            # print('u\n',vel_equation.dense[::3,::3])
            vel_equation.relax(self.u_relaxation_factor,model)
            # 
            # print('v\n',vel_equation.dense[0::3,0::3])
            # print('v\n',vel_equation.rhs.numpy()[::3])
            Ap = vel_equation.diagonal
            outer_loop_result,HbyA = vel_equation.solve_Axb(HbyA)

            p_grad = model.get_gradient('p').flatten()
            rUA = calculate_rUA(Ap,density,model.cells,rUA)
            HbyA = get_HbyA(HbyA,rUA,p_grad)

            rUA_faces = interpolate_cell_value_to_face(rUA_faces,rUA,model.faces)
            model.replace_cell_values(['u','v','w'],HbyA)
            
            for _ in range(self.NUM_INNER_LOOPS):
                
                model.face_interpolation()
                model.calculate_gradients()
                model.calculate_mass_flux()

                # Pressure Correction
                div_u = model.divFlux()
                
                p_correction_diffusion(model,viscosity = rUA_faces)

                p_corr_equation.form_system(p_correction_diffusion,fvm = model)
                p_corr_equation.add_RHS(div_u)

                if model.reference_pressure_cell_id is not None:
                    p_corr_equation.replace_row(model.reference_pressure_cell_id,model.reference_pressure)
                # print('p\n',p_corr_equation.dense)
                inner_loop_result,p_cor = p_corr_equation.solve_Axb()

                model.relax(p_cor,alpha = self.p_relaxation_factor, output_index= 3)
                
                model.face_interpolation('p')
                model.calculate_gradients('p')
                vel_correction = model.get_gradient('p',coeff = rUA).flatten()
                
                HbyA -= vel_correction
                # sub_1D_array(HbyA,vel_correction,HbyA)
                model.replace_cell_values(['u','v','w'],HbyA)
                

            if i % steps_per_check == 0:
                print(f'step iter: {i}')
                print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
                converged = model.check_convergence(vel_equation.matrix,HbyA,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
                
                if converged:
                    print(f'Run Reached Convergence Criteria at iteration {i}')
                    return None
                
        print(f'MAX ITERATIONS OF {num_steps} REACHED. Terminating')
        print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
        converged = model.check_convergence(vel_equation.matrix,HbyA,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
