import warp as wp
from warp_cfd.FV import FVM
import numpy as np
from warp_cfd.FV.terms import ConvectionTerm,DiffusionTerm, GradTerm,Matrix
from warp_cfd.FV.field import Field
import warp as wp
from warp_cfd.FV.Ops.array_ops import sub_1D_array

class SIMPLE():
    def __init__(self,model:FVM,u_relaxation_factor=0.7,p_relaxation_factor = 0.3) -> None:
        self.model = model
        self.convection = ConvectionTerm(model,model.fields[0:-1],'upwind') # We only want velocities
        self.diffusion = DiffusionTerm(model,model.fields[0:-1])
        self.grad_P = GradTerm(model,model.fields[-1]) # P

        self.vel_equation = Matrix(model,fields = model.fields[0:-1])

        self.p_correction_diffusion = DiffusionTerm(model,Field('p_cor'),von_neumann= 0., need_global_index= False)
        self.p_corr_equation = Matrix(model,fields = Field('p_cor'))

        self.vel_correction = wp.zeros(shape=(model.num_cells,3),dtype=float)

        self.vel_array = wp.zeros(shape=(model.num_cells*3),dtype=model.float_dtype)

        self.p_relaxation_factor = p_relaxation_factor
        self.u_relaxation_factor = u_relaxation_factor
        self.NUM_INNER_LOOPS = 1

    def run(self,num_steps,steps_per_check = 10,*,rhie_chow = True):
        model = self.model
        convection = self.convection
        diffusion = self.diffusion
        grad_P = self.grad_P
        p_correction_diffusion = self.p_correction_diffusion
        vel_equation= self.vel_equation
        p_corr_equation = self.p_corr_equation
        vel_array = self.vel_array

        for i in range(num_steps):
            model.face_interpolation()
            model.calculate_gradients()
            model.calculate_mass_flux(rhie_chow=rhie_chow)

            # intermediate Velocity Step
            convection(model)
            diffusion(model,viscosity = model.viscosity)
            grad_P(model)
            vel_equation.form_system([convection,-diffusion],explicit_terms= -grad_P,fvm = model)
            
            vel_equation.relax(self.u_relaxation_factor,model)
            # print(vel_equation.dense[::3,::3])
            ap = vel_equation.diagonal
            outer_loop_result,vel_array = vel_equation.solve_Axb(vel_array)
            # print(vel_array[::3])
            model.replace_cell_values([0,1,2],vel_array)

            model.pressure_correction_ops.calculate_D_viscosity(model.D_cell,model.D_face,ap,model.cells,model.faces)

            model.face_interpolation()
            model.calculate_gradients()
            model.calculate_mass_flux(rhie_chow=rhie_chow)

            # Pressure Correction
            div_u = model.calculate_divergence()
            
            p_correction_diffusion(model,viscosity = model.D_face)

            p_corr_equation.form_system(p_correction_diffusion,fvm = model)
            p_corr_equation.add_RHS(div_u)
            # print(p_corr_equation.dense)
            p_corr_equation.replace_row(0,0.)
            
            inner_loop_result,p_cor = p_corr_equation.solve_Axb()
            #Update Pressure
            model.update_cell_values(3,p_cor,scale = self.p_relaxation_factor)
            
            #Update Velocity
            vel_correction = p_corr_equation.calculate_gradient(coeff=model.D_face,fvm = model)
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
