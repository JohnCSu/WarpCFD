
from warp_cfd.FV.model import FVM
import numpy as np
import warp as wp
from warp_cfd.FV.implicit_Solvers import SIMPLE
from warp_cfd.preprocess import Mesh

from warp_cfd.preprocess import create_2D_grid,Mesh,define_boundary_walls
import pyvista as pv

wp.config.mode = "debug"
'''
LDC example for Re = 100 run for 2000 iterations for Hex mesh example. Here Othrogonal correctors are turned off as it is essentially a cartesian grid
'''
if __name__ == '__main__':
    wp.init()
    # wp.clear_kernel_cache()
    np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)
    n = 41# Approximate number of cells in x and y direction
    w,l = 1.,1.
    Re = 100
    G,nu = 1,1/Re
    dz =0.1
    pv_mesh = create_2D_grid((0,0,0), n, n , 1,1,dz = dz,element_type= 'hex',display_mesh= False,save = 'wedge')
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
    model.set_reference_pressure(0,0.)
    centroids = model.struct_member_to_array('centroid','cells')

    solver = SIMPLE(model,0.7,0.3,correction=False)
    solver.run(1000,100)

    # exit()
    from matplotlib import pyplot as plt

    velocity = solver.vel_array.numpy().reshape(-1,3)
    p = model.cell_values.numpy()[:,3]
    u = velocity[:,0]
    v = velocity[:,1]
    w = velocity[:,2]
    centroids = model.struct_member_to_array('centroid','cells')
    x,y,z = [centroids[:,i] for i in range(3)]

    pv_mesh['u'] = u
    pv_mesh['v'] = v
    pv_mesh['w'] = w
    pv_mesh['p'] = p
    pv_mesh['u_mag'] = np.sqrt(u**2 + v**2)

    # To save images set off_screen to False and uncomment screenshot line
    for output in ['u','v','p','u_mag']: 
        plotter = pv.Plotter(off_screen=False)
        plotter.add_mesh(pv_mesh,scalars = output, show_edges= False,n_colors= 100,cmap= 'jet')
        plotter.camera_position = 'xy'
        # plotter.screenshot(f'{output}.png')
        plotter.show()
        
    
    import pandas as pd
    v_benchmark = pd.read_csv(r'examples\LDC\v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv(r'examples\LDC\u_velocity_results.txt',sep= '\t')

    points = np.linspace(0.,1.,n)
    horizontal_centerline = pv.Line((0,0.5,dz/2),(1,0.5,dz/2),len(points)-1)
    horizontal_centerline= horizontal_centerline.sample(pv_mesh, pass_point_data=False)
    v_05 = horizontal_centerline['v']
    print(len(v_05))
    print(f"CFD max {v_05.max()}, Benchmark Max :{v_benchmark['100'].max()}")
    plt.plot(v_benchmark['%x'],v_benchmark[str(Re)],'o',label = 'Ghia et al')
    
    
    plt.plot(points,v_05,label = 'CFD Code')
    plt.legend()
    # plt.savefig('v_velocity at hori_centerline.png')
    plt.show()

    vertical_centerline = pv.Line((0.5,0.,dz/2),(0.5,1.,dz/2),len(points)-1)
    vertical_centerline= vertical_centerline.sample(pv_mesh, pass_point_data=False)

    u_05 = vertical_centerline['u']
    print(f"CFD max {u_05.max()}, Benchmark Max :{u_benchmark['100'].max()}")
    plt.plot(u_benchmark['%y'],u_benchmark[str(Re)],'o',label = 'Ghia et al')
    plt.plot(points,u_05,label = 'CFD Code')
    plt.legend()
    # plt.savefig('u_velocity at vert_centerline.png')
    plt.show()
