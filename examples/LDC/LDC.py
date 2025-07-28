
from warp_cfd.FV.model import FVM
import numpy as np
import warp as wp
from warp_cfd.FV.Ops.array_ops import sub_1D_array
from warp_cfd.FV.implicit_Solvers import SIMPLE
from warp_cfd.preprocess import Mesh

from warp_cfd.preprocess import create_2D_grid,Mesh,define_boundary_walls
wp.config.mode = "debug"
wp.init()
if __name__ == '__main__':
    from grid import create_hex_grid
    # wp.clear_kernel_cache()
    np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)
    n = 41
    w,l = 1.,1.
    Re = 100
    G,nu = 1,1/Re
    pv_mesh = create_2D_grid((0,0,0), n, n , 1,1,element_type= 'hex',display_mesh= False,save = 'wedge')
    m = Mesh(pv_mesh,num_outputs=4)
    define_boundary_walls(m)
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
    
    model = FVM(m,output_variables = ['u','v','w','p'],density = 1.,viscosity= nu,float_dtype =wp.float32)
    model.init_step()
    centroids = model.struct_member_to_array('centroid','cells')
    results = m.pyvista_mesh
    IC = np.ones(shape = (model.num_cells,model.num_outputs),dtype= np.float32)
    IC[-1,:] = 0.
    # model.set_initial_conditions(wp.array(IC))

    solver = SIMPLE(model,0.7,0.3,correction=True)
    solver.run(2000,100)

    # exit()
    from matplotlib import pyplot as plt

    velocity = solver.vel_array.numpy().reshape(-1,3)
    p = model.cell_values.numpy()[:,3]
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
    
    plt.tricontourf(x,y,p,cmap ='jet',levels = 100)
    plt.colorbar()
    plt.show()


    plt.tricontourf(x,y,np.sqrt(u**2 + v**2),cmap ='jet',levels = np.linspace(0,1,100,endpoint= True))
    plt.colorbar()
    plt.show()


    import pandas as pd
    v_benchmark = pd.read_csv(r'examples\LDC\v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples\LDC\u_velocity_results.txt',sep= '\t')

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