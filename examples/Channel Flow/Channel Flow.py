
from warp_cfd.FV.model import FVM
import numpy as np
import warp as wp
from warp_cfd.FV.implicit_Solvers import IncompressibleSolver
from warp_cfd.preprocess import Mesh

from warp_cfd.preprocess import create_2D_grid,Mesh,define_boundary_walls
import pyvista as pv

wp.config.mode = "debug"
'''
Channel Flow for Low Re = 1. Here We test the pressure inlet and outlets conditions

As the Re is increased, there is more inertia, so it is more difficult to reach convergence with presure inlets and so a lower relaxation factors should be used

'''
if __name__ == '__main__':
    wp.init()
    # wp.clear_kernel_cache()
    np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)
    
    width,height = 5.,1.
    ny = 10
    nx = ny*int(width)# Approximate number of cells in x and y direction
    
    Re = 100
    G,nu = 1,1/Re
    dz =0.1
    pv_mesh = create_2D_grid((0,0,0), nx, ny , width,height,dz = dz,element_type= 'hex',display_mesh= False,save = 'wedge')
    m = Mesh(pv_mesh,num_outputs=4)
    define_boundary_walls(m)
    # IC = np.load(f'benchmark_n{n}.npy')
    
    
    # m.set_cell_value(0,p= 0)
    
    model = FVM(m,output_variables = ['u','v','w','p'],density = 1.,viscosity= nu,float_dtype =wp.float32)


    model.boundary.pressure_BC('+X', p = 0)
    model.boundary.no_slip_wall('-Y')
    model.boundary.no_slip_wall('+Y')
    model.boundary.pressure_BC('-X',p = 1)
    model.boundary.slip_wall('-Z')
    model.boundary.slip_wall('+Z')
    model.initialize()
    centroids = model.struct_member_to_array('centroid','cells')

    solver = IncompressibleSolver(model,0.1,0.1,correction=False)
    solver.run(1000,100)

    # exit()
    from matplotlib import pyplot as plt

    velocity = model.cell_values.numpy()[:,0:3] 
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
        
    h_points = np.linspace(0.,width,20)
    horizontal_centerline = pv.Line( (0,height/2,dz/2) , (width,height/2,dz/2) ,len(h_points)-1)
    horizontal_centerline= horizontal_centerline.sample(pv_mesh, pass_point_data=False)
    v_05 = horizontal_centerline['v']

    plt.plot(h_points,v_05,label = 'CFD Code')
    plt.legend()
    plt.show()


    v_points = np.linspace(0.,height,ny)
    vertical_centerline = pv.Line((width/2,0.,dz/2),(width/2,height,dz/2),len(v_points)-1)
    vertical_centerline= vertical_centerline.sample(pv_mesh, pass_point_data=False)

    u_05 = vertical_centerline['u']
    plt.plot(u_05,v_points,label = 'CFD Code')
    plt.legend()
    # plt.savefig('u_velocity at vert_centerline.png')
    plt.show()
