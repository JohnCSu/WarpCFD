import warp as wp
from warp_cfd.FV import FVM
import numpy as np
from warp_cfd.FV.terms import ConvectionTerm,DiffusionTerm, GradTerm,Matrix
from warp_cfd.FV.field import Field
import warp as wp
from warp_cfd.FV.Ops.array_ops import sub_1D_array,add_1D_array,div_1D_array
from warp_cfd.FV.Ops.fv_ops import interpolate_cell_value_to_face,calculate_rUA
class SIMPLE():
    def __init__(self,model:FVM,u_relaxation_factor=0.7,p_relaxation_factor = 0.3,correction = False) -> None:
        self.model = model

        self.float_dtype = model.float_dtype
        velocity_vars = ['u','v','w']
        self.convection = ConvectionTerm(model,velocity_vars,'upwind') # We only want velocities
        self.diffusion = DiffusionTerm(model,velocity_vars,correction = correction)
        self.grad_P = GradTerm(model,'p') # P

        self.vel_equation = Matrix(model,fields = velocity_vars)

        self.p_correction_diffusion = DiffusionTerm(model,'p_cor',need_global_index= True,von_neumann= 0.,correction=correction)
        self.p_corr_equation = Matrix(model,fields = 'p_cor')

        self.vel_correction = wp.zeros(shape=(model.num_cells,3),dtype=float)

        self.vel_array = wp.zeros(shape=(model.num_cells*3),dtype=model.float_dtype)

        self.p_relaxation_factor = p_relaxation_factor
        self.u_relaxation_factor = u_relaxation_factor
        self.NUM_INNER_LOOPS = 3

        self.HbyA = wp.zeros_like(self.vel_array)
        self.grad_P_HbyA = wp.zeros_like(self.vel_array)
        self.rUA = wp.zeros(shape= model.cells.shape[0],dtype= model.float_dtype)
        self.rUA_faces = wp.zeros(shape= model.faces.shape[0],dtype= model.float_dtype)

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
            model.face_interpolation()
            
            model.calculate_gradients()
            model.calculate_mass_flux(rhie_chow=rhie_chow)

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
            
            
            model.replace_cell_values([0,1,2],vel_array)
            

            rUA = calculate_rUA(Ap,model.cells,rUA)
            rUA_faces = interpolate_cell_value_to_face(rUA_faces,rUA,model.faces)


            # model.pressure_correction_ops.calculate_D_viscosity(model.D_cell,model.D_face,Ap,model.cells,model.faces)

            # grad_P_HbyA = div_1D_array(grad_P.weights,Ap,grad_P_HbyA)
            # HbyA = add_1D_array(vel_array,grad_P_HbyA,HbyA)
            # Interpolate HbyA to Faces and get mass flux
            # Add to RHS

            for _ in range(self.NUM_INNER_LOOPS):
                model.face_interpolation()
                model.calculate_gradients()
                model.calculate_mass_flux(rhie_chow=rhie_chow)

                # Pressure Correction
                div_u = model.calculate_divergence()
                
                p_correction_diffusion(model,viscosity = rUA_faces)

                p_corr_equation.form_system(p_correction_diffusion,fvm = model)
                p_corr_equation.add_RHS(div_u)
                # print(p_corr_equation.dense)
                p_corr_equation.replace_row(0,0.)
                
                inner_loop_result,p_cor = p_corr_equation.solve_Axb()
                #Update Pressure
            #     model.replace_cell_values(4,p_cor)
                # print(model.cell_values.numpy()[:,3])
                model.update_cell_values(3,p_cor,scale = self.p_relaxation_factor)
                # print(model.cell_values.numpy()[:,3])
                #Update Velocity
                vel_correction = p_corr_equation.calculate_gradient(coeff=rUA_faces,fvm = model)
                sub_1D_array(vel_array,vel_correction.flatten(),vel_array)
                model.replace_cell_values([0,1,2],vel_array)
            


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
