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
        
        if not model.finalized:
            raise ValueError('Please call method model.finalize() before passing model in to solver')
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


        self.set_time_stepping(1,1.)
        self.NUM_ORTHOGONAL_CORRECTORS = 1
        self.NUM_OUTER_LOOPS = 1
        self.NUM_INNER_LOOPS = 1

    def set_time_stepping(self,MAX_TIME_STEPS,dt,automatic = False,CFL_Max = 1.):
        '''
        Lets assume constant dt
        '''
        if automatic is False:
            self.MAX_TIME_STEPS = MAX_TIME_STEPS
            self.dt = dt
        else:
            raise NotImplementedError('automatic time stepping not yet implemented')


    def set_iterations(self,NUM_OUTER_LOOPS,NUM_INNER_LOOPS,NUM_ORTHOGONAL_CORRECTORS):
        self.NUM_OUTER_LOOPS = NUM_OUTER_LOOPS
        self.NUM_INNER_LOOPS = NUM_INNER_LOOPS
        self.NUM_ORTHOGONAL_CORRECTORS = NUM_ORTHOGONAL_CORRECTORS


    def run(self,num_steps,steps_per_check = 10):
        model = self.model

        model.finalize()
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

        self.NUM_OUTER_LOOPS = num_steps

        for nt in range(self.MAX_TIME_STEPS):
            for nOuter_i in range(self.NUM_OUTER_LOOPS):
                model.set_boundary_conditions()
                model.face_interpolation()
                model.calculate_gradients()
                model.calculate_mass_flux()

                # intermediate Velocity Step
                convection(model)
                diffusion(model,viscosity = kinematic_viscosity)
                grad_P(model)
                vel_equation.form_system([convection,-diffusion],explicit_terms= -inv_density*grad_P,fvm = model)
                
                vel_equation.relax(self.u_relaxation_factor,model)
                
                Ap = vel_equation.diagonal
                outer_loop_result,HbyA = vel_equation.solve_Axb(HbyA)

                p_grad = model.get_gradient('p').flatten()
                rUA = calculate_rUA(Ap,density,model.cells,rUA)
                HbyA = get_HbyA(HbyA,rUA,p_grad)

                rUA_faces = interpolate_cell_value_to_face(rUA_faces,rUA,model.faces)
                model.replace_cell_values(['u','v','w'],HbyA)
                
                for nInner_i in range(self.NUM_INNER_LOOPS):
                    for nOrtho_i in range(self.NUM_ORTHOGONAL_CORRECTORS):
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
                    

                if nOuter_i % steps_per_check == 0:
                    print(f'step iter: {nOuter_i}')
                    print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
                    converged = model.check_convergence(vel_equation.matrix,HbyA,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
                    
                    if converged:
                        print(f'Run Reached Convergence Criteria at iteration {i}')
                        return None
                    
        print(f'MAX ITERATIONS OF {num_steps} REACHED. Terminating')
        print(f'Outer loop Linear Solve {outer_loop_result} Inner loop solve {inner_loop_result}:')
        converged = model.check_convergence(vel_equation.matrix,HbyA,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
