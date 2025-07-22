
from warp_cfd.FV.model import FVM
import numpy as np
from warp_cfd.FV.terms.convection import ConvectionTerm
from warp_cfd.FV.terms.diffusion import DiffusionTerm
from warp_cfd.FV.terms.grad import GradTerm
from warp_cfd.FV.field import Field
from warp_cfd.FV.terms.matrix import Matrix
import warp as wp
from warp_cfd.FV.Ops.array_ops import sub_1D_array
from warp_cfd.FV.implicit_Solvers import SIMPLE

wp.config.mode = "debug"
wp.init()
if __name__ == '__main__':
    from grid import create_hex_grid
    # wp.clear_kernel_cache()
    n = 41
    w,l = 1.,1.
    Re = 100
    G,nu = 1,1/Re
    m = create_hex_grid(n,n,1,(w/n,l/n,0.1))

    # IC = np.load(f'benchmark_n{n}.npy')
    m.set_boundary_value('+X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-Y',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('+Y',u = 1,v = 0,w = 0) # Velocity Inlet

    '''
    Add check that All bf have some fixed value => Boundary IDs should equal same length as boundary faces
    '''
    m.set_gradient_value('-Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+X',p = 0) # No Slip
    m.set_gradient_value('-X',p = 0) # No Slip
    m.set_gradient_value('+Y',p = 0) # Velocity Inlet
    m.set_gradient_value('-Y',p = 0) # Velocity Inlet
    
    m.set_cell_value(0,p= 0)
    
    model = FVM(m,density = 1.,viscosity= nu)
    model.init_step()
    results = m.pyvista_mesh
    
    solver = SIMPLE(model)
    solver.run(500,40)



    # convection = ConvectionTerm(model,model.fields[0:-1],'upwindLinear') # We only want velocities
    # diffusion = DiffusionTerm(model,model.fields[0:-1])
    # grad_P = gradTerm(model,model.fields[-1]) # P

    # vel_equation = Matrix(model,fields = model.fields[0:-1])

    # p_correction_diffusion = DiffusionTerm(model,Field('p_cor'),von_neumann= 0., need_global_index= False)
    # p_corr_equation = Matrix(model,fields = Field('p_cor'))

    # np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)

    # vel_correction = wp.zeros(shape=(model.num_cells,3),dtype=float)

    # vel_array = wp.zeros(shape=(model.num_cells*3),dtype=model.float_dtype)

    # # IC = wp.ones(shape = (model.num_cells,3),dtype = wp.float32)
    # # model.set_initial_conditions(IC)

    # from warp_cfd.FV.utils import bsr_to_coo_array
    # # model.MAX_STEPS = 500
    # for i in range(500):
    #     model.face_interpolation()
    #     model.calculate_gradients()
    #     model.calculate_mass_flux(rhie_chow=True)

    #     # intermediate Velocity Step
    #     convection(model)
    #     diffusion(model,viscosity = nu)
    #     grad_P(model)
    #     vel_equation.form_system([convection,-diffusion],explicit_terms= -grad_P,fvm = model)
        
    #     vel_equation.relax(0.7,model)
    #     ap = vel_equation.diagonal
    #     outer_loop_result,vel_array = vel_equation.solve_Axb(vel_array)
        
    #     model.replace_cell_values([0,1,2],vel_array)

    #     model.pressure_correction_ops.calculate_D_viscosity(model.D_cell,model.D_face,ap,model.cells,model.faces)

    #     model.face_interpolation()
    #     model.calculate_gradients()
    #     model.calculate_mass_flux(rhie_chow=True)

    #     # Pressure Correction
    #     div_u = model.calculate_divergence()
        
    #     p_correction_diffusion(model,viscosity = model.D_face)

    #     p_corr_equation.form_system(p_correction_diffusion,fvm = model)
    #     p_corr_equation.add_RHS(div_u)
    #     p_corr_equation.replace_row(0,0.)

    #     inner_loop_result,p_cor = p_corr_equation.solve_Axb()
    #     #Update Pressure
    #     model.update_cell_values(3,p_cor,scale = 0.3)
        
    #     #Update Velocity
    #     vel_correction = p_corr_equation.calculate_gradient(coeff=model.D_face,fvm = model)
    #     sub_1D_array(vel_array,vel_correction.flatten(),vel_array)
    #     model.replace_cell_values([0,1,2],vel_array)
        
    #     if model.steps % 40 == 0:
    #         print(f'step iter: {model.steps}')
    #         converged = model.check_convergence(vel_equation.matrix,vel_array,vel_equation.rhs,div_u,vel_correction.flatten(),p_cor)
    #         # print(vel_result)

    # exit()
    from matplotlib import pyplot as plt

    velocity = solver.vel_array.numpy().reshape(-1,3)
    p = model.cell_values.numpy()[:,-1]
    u = velocity[:,0]
    v = velocity[:,1]
    w = velocity[:,2]
    centroids = model.struct_member_to_array('centroid','cells')
    x,y,z = [centroids[:,i] for i in range(3)]

    plt.tricontourf(x,y,u,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()

    plt.tricontourf(x,y,v,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()

    plt.tricontourf(x,y,np.sqrt(u**2 + v**2),cmap ='jet',levels = np.linspace(0,1,100,endpoint= True))
    plt.colorbar()
    plt.show()

    import pandas as pd
    v_benchmark = pd.read_csv('v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv('u_velocity_results.txt',sep= '\t')

    v_05 = v[y == 0.5]
    print(f'CFD max {v_05.max()}, Benchmark Max :{v_benchmark['100'].max()}')
    plt.plot(v_benchmark['%x'],v_benchmark[str(Re)],'o',label = 'Ghia et al')
    plt.plot(x[y == 0.5],v_05,label = 'CFD Code')
    plt.legend()
    plt.show()

    u_05 = u[x == 0.5]
    print(f'CFD max {v_05.max()}, Benchmark Max :{u_benchmark['100'].max()}')
    plt.plot(u_benchmark['%y'],u_benchmark[str(Re)],'o',label = 'Ghia et al')
    plt.plot(y[x == 0.5],u_05,label = 'CFD Code')
    plt.legend()
    plt.show()


    
        # model.intermediate_velocity_step.solve()
    # exit()
    # print(model.mass_fluxes.numpy())