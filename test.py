
from warp_cfd.FV.finiteVolume import FVM
import numpy as np
from warp_cfd.FV.terms.convection import ConvectionTerm
from warp_cfd.FV.terms.diffusion import DiffusionTerm
from warp_cfd.FV.terms.field import Field
import warp as wp
wp.config.mode = "debug"
wp.init()
if __name__ == '__main__':
    from grid import create_hex_grid
    # wp.clear_kernel_cache()
    n = 51
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
    # m.set_gradient_value('-Y',u=0,v=0,w=0) # No Slip
    m.set_gradient_value('+Y',p = 0) # Velocity Inlet
    m.set_gradient_value('-Y',p = 0) # Velocity Inlet
    
    m.set_cell_value(0,p= 0)
    
    model = FVM(m,density = 1.,viscosity= nu)
    model.init_step()

    # IC = wp.ones(shape = (model.num_cells,3),dtype = wp.float32)
    # model.set_initial_conditions(IC)
    results = m.pyvista_mesh
    
    convection = ConvectionTerm(model,model.fields[0:-1],'upwindLinear') # We only want velocities
    diffusion = DiffusionTerm(model,model.fields[0:-1])

    p_correction_diffusion = DiffusionTerm(model,model.fields[-1],von_neumann= 0.)

    np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)
    from warp_cfd.FV.utils import bsr_to_coo_array
    model.MAX_STEPS = 500
    for i in range(model.MAX_STEPS):
        model.face_interpolation()
        model.calculate_gradients()
        model.calculate_mass_flux(rhie_chow=True)
        # print(model.face_properties.boundary_face_ids)
        # print(model.face_properties.internal_face_ids)
        convection(model)
        diffusion(model,viscosity = nu)

        vel_result,vel_matrix,b,Ap = model.intermediate_velocity_step.solve(model.initial_velocity,
                                                                   model.intermediate_velocity,
                                                                   model.cell_values,
                                                                   model.cell_gradients,
                                                                   model.cells,model.faces,
                                                                   diffusion.weights,
                                                                   convection.weights,
                                                                   model.density,
                                                                   model.vel_indices)
        
        model.pressure_correction_ops.calculate_D_viscosity(model.D_cell,model.D_face,Ap,model.cells,model.faces)
        p_correction_diffusion(model,model.D_face)
        # print(vel_result)
        # print(bsr_to_coo_array(vel_matrix).toarray())
        # print(diffusion.weights.numpy())
        # print(convection.weights.numpy())
        # print(p_correction_diffusion.weights.numpy())
        for _ in range(model.NUM_INNER_LOOPS):
            model.face_interpolation()
            model.calculate_gradients()
            model.calculate_mass_flux(True)
            # print(model.D_face)
            # 
            p,div_u,p_correction,velocity_correction = model.pressure_correction_step.solve(p_correction_diffusion.weights,
                                                                                            model.intermediate_velocity,
                                                                                            model.corrected_velocity,
                                                                                            model.cell_values,
                                                                                            model.D_face,
                                                                                            model.mass_fluxes,
                                                                                            model.cells,
                                                                                            model.faces)
            if model.num_cells > 1: # We only need to reinitialise intermediate velocity if multiple inner loops are run as this
                wp.copy(model.intermediate_velocity,model.corrected_velocity)


        wp.copy(model.initial_velocity,model.corrected_velocity)

        if model.steps % 40 == 0:
            print(f'step iter: {model.steps}')
            converged = model.check_convergence(vel_matrix,model.corrected_velocity,b,div_u,velocity_correction,p_correction)
            print(vel_result)
        model.steps += 1 
    from matplotlib import pyplot as plt

    velocity = model.corrected_velocity.numpy().reshape(-1,3)
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