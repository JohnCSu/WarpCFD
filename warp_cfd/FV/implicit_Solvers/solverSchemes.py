from .incompressible import IncompressibleSolver
from warp_cfd.FV import FVM

def SIMPLE(model:FVM,num_outer_loops,u_relaxation_factor = 0.7,p_relaxation_factor = 0.3,orthogonal_Correction = False,num_inner_loops = 1,num_orthogonal_corrector_loops = 1,):
    model.finalize()
    solver = IncompressibleSolver(model,u_relaxation_factor,p_relaxation_factor,correction=orthogonal_Correction,steady_state= True)
    solver.set_iterations(num_outer_loops,num_inner_loops,num_orthogonal_corrector_loops)
    return solver

def PISO(model:FVM,dt,max_time_steps,orthogonal_Correction = False,num_outer_loops = 1,num_inner_loops = 3,num_orthogonal_corrector_loops = 1):
    model.finalize()
    solver = IncompressibleSolver(model,correction=orthogonal_Correction,steady_state= False)
    
    solver.set_time_stepping(max_time_steps,dt)
    solver.set_iterations(num_outer_loops,num_inner_loops,num_orthogonal_corrector_loops)
    return solver

