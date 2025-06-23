import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp.types import vector
from warp import sparse
from warp.optim import linear
from typing import Any

import scipy.sparse as sp_sparse
from warp_cfd.FV.Ops.array_ops import add_1D_array,sub_1D_array,inv_1D_array,to_vector_array,mult_scalar_1D_array,absolute_1D_array,power_scalar_1D_array,sum_1D_array,L_norm_1D_array

from matplotlib import pyplot as plt
from warp_cfd.preprocess import Mesh
import warp_cfd.FV.Cells as Cells
from warp_cfd.FV.Weights import create_weight_struct
from warp_cfd.FV.Ops import Mesh_Ops,Weights_Ops,Matrix_Ops,Pressure_correction_Ops 
from warp_cfd.FV.Convergence import Convergence
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.intermediate_velocity import intermediate_velocity_step
from warp_cfd.FV.pressure_correction import pressure_correction_step
wp.config.mode = "debug"
# wp.config.verify_fp = True
'''
TODO:
- Form Sparse Matrix
- implement simple Algo
TO ADD LATER:
- Add non-orthoganality for laplacian (Currently no non-ortho)
- Add upwind for convection and options (currently only central differencing)
- Add least squares and node based gauss for gradient calcs

FUTURE
- Time Dependent NS
- Add RANS
- Add LES

'''
class CFD_settings:
    steady_state: bool = False
    face_interpolation: int = 'upwind-linear'

class FVM():
    def __init__(self,mesh:Mesh,density:float = 1000,viscosity:float = 1e-3,dtype = wp.float32,int_dtype = wp.int32):
        self.density = density
        self.viscosity = viscosity
        self.gridType:str = mesh.gridType
        self.dimension:int = mesh.dimension
        self.float_dtype = dtype
        self.int_dtype = int_dtype
        self.cellType = mesh.cellType

        self.settings = CFD_settings()
        self.Convergence = Convergence()

        self.face_properties = mesh.face_properties.to_NVD_warp()
        self.cell_properties = mesh.cell_properties.to_NVD_warp()
        self.node_properties = to_vector_array(mesh.nodes)        
        # Output for each cell: T,C,4 (u,v,w,p)
        self.input_variables = ['x','y','z','t']
        self.output_variables = ['u','v','w','p']
        self.vars_mapping = {var:i for i,var in enumerate(self.output_variables)}
        self.vel_indices = wp.array([0,1,2],dtype=self.int_dtype)
        self.p_index = wp.array([3],dtype=self.int_dtype)
        self.num_outputs = len(self.output_variables)

        self.num_nodes = mesh.nodes.shape[0]
        self.num_cells = mesh.cells.shape[0]
        self.num_faces = mesh.face_properties.unique_faces.shape[0]
        
        self.nodes_per_cell = mesh.cells.shape[-1]
        self.faces_per_cell = self.cell_properties.faces_per_cell
        self.nodes_per_face = self.cell_properties.nodes_per_face

        # #Field Outputs
        # self.field_output_vector = vector(length = len(self.output_variables),dtype= self.float_dtype)
        # self.field_output = wp.zeros(shape = (self.num_cells),dtype=self.field_output_vector)
        # self.field_output_gradient_matrix = wp.mat(shape = (len(self.output_variables),3),dtype= self.float_dtype)
        # self.field_output_gradients = wp.zeros(shape = self.num_cells,dtype=self.field_output_gradient_matrix)
        
        self.cell_struct,self.face_struct,self.node_struct= Cells.create_mesh_structs(self.nodes_per_cell,self.faces_per_cell,self.nodes_per_face,self.num_outputs,self.dimension,self.float_dtype,self.int_dtype)
        self.weight_struct = create_weight_struct(self.float_dtype)
        self.cells = wp.zeros(shape=self.num_cells,dtype= self.cell_struct)
        '''Array of cell structs for storing information about cell dependent values'''
        self.faces = wp.zeros(shape = self.num_faces,dtype=self.face_struct)
        '''Array of face structs for storing information about face dependent values'''
        self.nodes = wp.zeros(shape= self.num_nodes,dtype=self.node_struct)
        '''Array of node structs for storing inforation about nodes (mainly id and coordinates)'''
        self.laplacian_weights = wp.zeros(shape = (self.num_cells,self.faces_per_cell,self.num_outputs),dtype= self.weight_struct)
        '''Array of structs of size (C,F,O) Store Cells Struct which contains cell info'''
        self.convection_weights = wp.zeros(shape = (self.num_cells,self.faces_per_cell,self.num_outputs),dtype= self.weight_struct)

        self.face_viscosity = None
        
        self.steps = 0
        self.MAX_STEPS = 2

        Ops_args = [self.cell_struct,self.face_struct,self.node_struct,self.weight_struct,self.cell_properties,self.face_properties,self.num_outputs,self.float_dtype,self.int_dtype]

        self.mesh_ops = Mesh_Ops(*Ops_args)
        self.weight_ops = Weights_Ops(*Ops_args) 
        self.matrix_ops = Matrix_Ops(*Ops_args)
        self.pressure_correction_ops = Pressure_correction_Ops(*Ops_args)
    
    def init_step(self):
        '''
        Initialises The Model. For now assume that the mesh and BC are both Fixed in Time
        '''
        Cells.init_structs(self.cells,self.faces,self.nodes,self.cell_properties,self.face_properties,self.node_properties)
        self.mesh_ops.init()
        self.weight_ops.init()
        self.matrix_ops.init()
        self.pressure_correction_ops.init()
        self.intermediate_velocity_step = intermediate_velocity_step(self.mesh_ops,self.weight_ops,self.matrix_ops)
        self.pressure_correction_step = pressure_correction_step(self.mesh_ops,self.weight_ops,self.matrix_ops,self.pressure_correction_ops)
        
        self.intermediate_velocity_step.init(self.cells,self.faces)
        self.pressure_correction_step.init(self.cells,self.faces)

        self.init_global_arrays()
    

    def set_initial_conditions(self,IC):
        self.mesh_ops.set_initial_conditions(IC,self.cells)

    def init_global_arrays(self):
        self.massflux_array = wp.empty(shape = (self.num_cells,self.faces_per_cell))
        
        self.intermediate_velocity = wp.zeros(shape = self.num_cells*self.dimension,dtype=self.float_dtype)
        self.initial_velocity = wp.zeros_like(self.intermediate_velocity)
        self.corrected_velocity = wp.zeros_like(self.intermediate_velocity)
        
        self.gradients = wp.zeros(shape = (self.num_cells,self.num_outputs),dtype = self.mesh_ops.vector_type)
        self.D_face = wp.zeros(shape = (self.num_faces),dtype=self.float_dtype)
        self.D_cell = wp.zeros(shape = self.num_cells,dtype = self.mesh_ops.vector_type)

    def update_flux(self,rhie_chow=True):
        self.gradients.zero_() # Use it to store stuff
        self.mesh_ops.apply_BC(self.faces)
        # self.mesh_ops.apply_cell_value(self.cells)
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.p_index,interpolation_method=0)
        # print('p face',self.struct_member_to_array('values','faces')[:,-1])
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.vel_indices,interpolation_method=0)
        
        self.mesh_ops.calculate_gradients(self.gradients,self.cells,self.faces,self.nodes,self.p_index)
        if rhie_chow:
            self.mesh_ops.rhie_chow_correction(self.gradients,self.D_face,self.cells,self.faces,self.vel_indices)
        
        self.mesh_ops.calculate_gradients(self.gradients,self.cells,self.faces,self.nodes,self.vel_indices)
        # print('g',self.gradients.numpy()[:,-1])
        self.mesh_ops.gradient_to_cell_struct(self.gradients,self.cells)

        self.mesh_ops.calculate_mass_flux(self.cells,self.faces) # Update Mass Flux
       
    def update_weights(self):
        # self.weight_ops.calculate_convection(self.cells,self.faces,self.convection_weights,output_indices= self.vel_indices)
        self.weight_ops.calculate_laplacian(self.cells,self.faces,self.laplacian_weights,output_indices= self.vel_indices,viscosity=self.viscosity/self.density)

    def check_convergence(self,vel_matrix:sparse.BsrMatrix,velocity:wp.array,b:wp.array,div_u=None,velocity_correction:wp.array=None,p_correction:wp.array=None,log = True):
        '''
        Checks the following:
        divergence residual
        momentum equations
        pressure correction residual
        '''
        
        self.mesh_ops.massflux_to_array(self.massflux_array,self.cells)
        if div_u is None:
            div_u = wp.zeros(shape = self.num_cells,dtype= self.float_dtype)
        self.mesh_ops.calculate_divergence(div_u,self.cells)
        Ax:wp.array = vel_matrix @ velocity

        Ax = Ax.numpy().reshape(-1,3)

        self.matrix_ops.form_p_grad_vector(self.intermediate_velocity_step.grad_P,self.cells,self.density)
        self.matrix_ops.get_b_vector(self.intermediate_velocity_step.H,self.intermediate_velocity_step.grad_P,b)

        b = b.numpy().reshape(-1,3)
        # print(Ax[:,1],b[:,1])
        convergence = {
            'momentum_x':self.Convergence.check('momentum_x',Ax[:,0],b[:,0]),
            'momentum_y':self.Convergence.check('momentum_y',Ax[:,1],b[:,1]),
            'momentum_z':self.Convergence.check('momentum_z',Ax[:,2],b[:,2]),
            'continuity':self.Convergence.check('continuity',div_u,self.massflux_array.flatten()) if div_u is not None else np.nan,
            'velocity_correction':self.Convergence.check('velocity_correction',velocity_correction) if velocity_correction is not None else np.nan,
            'pressure_correction':self.Convergence.check('pressure_correction',p_correction) if p_correction is not None else np.nan,
        }
        
        print(convergence)
        if log:
            self.Convergence.log(convergence)
        return self.Convergence.has_converged()
    
    
    
    def struct_member_to_array(self,member = 'mass_fluxes',struct = 'cells'):
        if struct == 'cells':
            struct_ = self.cell_struct
            arr = self.cells.numpy()
        elif struct == 'faces':
            struct_ = self.face_struct
            arr = self.faces.numpy()
        else:
            raise ValueError()
        keys = list(struct_.vars.keys())
        index = keys.index(member)

        member_arr = [a[index] for a in arr]
        return np.array(member_arr)

    def step(self):
        if self.steps == 0:
            self.update_flux(False)
            self.update_weights()

        vel_matrix,b,inv_A = self.intermediate_velocity_step.solve(self.initial_velocity,self.intermediate_velocity,self.cells,self.faces,self.laplacian_weights,self.convection_weights,self.density,self.vel_indices)
        # print(bsr_to_coo_array(vel_matrix).toarray())
        self.check_convergence(vel_matrix,self.intermediate_velocity,b,log= False)
        self.pressure_correction_ops.calculate_D_viscosity(self.D_cell,self.D_face,inv_A,self.cells,self.faces)
        # print(self.D_face.numpy().reshape(-1,3))
        # print(inv_A.numpy().reshape(-1,3))
        self.update_flux()
        p,div_u,p_correction,velocity_correction = self.pressure_correction_step.solve(self.intermediate_velocity,self.corrected_velocity,self.D_face,self.cells,self.faces) 
        converged = self.check_convergence(vel_matrix,self.corrected_velocity,b,div_u,velocity_correction,p_correction)
        self.update_flux()
       
        self.update_weights()

        
        wp.copy(self.initial_velocity,self.corrected_velocity)
        self.steps += 1
        return converged



def bsr_to_coo_array(bsr:sparse.BsrMatrix):
    return sp_sparse.coo_array((bsr.values.numpy(),(bsr.uncompress_rows().numpy(),bsr.columns.numpy())),shape = (bsr.nrow,bsr.ncol))

from copy import deepcopy

def pouiselle_flow(xyz,width,G = 1.,nu = 0.01):
    '''
    Let it be from Y+ to Y- with dp/dy = 1

    ==>P(y) = y
    ==> u(y) = 0
    --> v(y) = G/(2*nu)*y(h-y)

    '''

    y = xyz[:,1]
    x = xyz[:,0]
    P = G*y # G*y
    u = np.zeros_like(y)
    
    v = G/(2*nu) * x*(x - width)
    w = np.zeros_like(y)
    return np.stack([u,v,w,P],axis = -1)



if __name__ == '__main__':
    from grid import create_hex_grid
    n = 21
    w,l = 1.,1
    G,nu = 1,1
    m = create_hex_grid(n,n,1,(w/n,l/n,0.1))

    IC = pouiselle_flow(m.cell_centroids,width= w,G = G, nu = nu)
    x = m.face_properties.centroid[m.groups['+Y'].ids][:,0] # x co of inlet
    inlet = G/(2*nu) * x*(x - w)
    plt.plot(x,inlet)
    plt.show()
    m.set_boundary_value('+X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-Y',u = 0,v = inlet,w = 0) # No Slip
    m.set_boundary_value('+Y',u = 0,v = inlet,w = 0) # Velocity Inlet
    # m.set_boundary_value('-Y',p = 0.) # No Slip
    # m.set_boundary_value('+Y',p = 1.) # Velocity Inlet


    m.set_gradient_value('-Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+X',p = 0) # No Slip
    m.set_gradient_value('-X',p = 0) # No Slip
    # m.set_gradient_value('-Y',p = 0) # No Slip
    # m.set_gradient_value('+Y',p = 0) # Velocity Inlet
    
    m.set_cell_value(0,p= 0)

    # print(m.nodes)
    
    model = FVM(m,density = 1.,viscosity= nu)
    model.init_step()
    # model.set_initial_conditions(wp.array(IC))
    
    
    results = m.pyvista_mesh
    # print(model.p.shape)
    results.cell_data['p'] = IC[:,-1]
    results.cell_data['u'] = IC[:,0]
    results.cell_data['v'] = IC[:,1]
    # results.plot(scalars="u", cmap="jet", show_scalar_bar=True)
    # results.plot(scalars="p", cmap="jet", show_scalar_bar=True)
    # results.plot(scalars="v", cmap="jet", show_scalar_bar=True)
    
    
    
    
    
    
    
    np.set_printoptions(linewidth=300,threshold=100000,precision = 5)
    model.MAX_STEPS = 2
    for i in range(model.MAX_STEPS):
        print(model.step())
        
    
    
    a = model.intermediate_velocity_step
    b = model.pressure_correction_step
    # dense = bsr_to_coo_array(b.p_correction_matrix).toarray()
    # print(dense)
   
    # print('intermediate vel',model.intermediate_velocity.numpy().max())
    # print(b.velocity_correction)
    # print(model.corrected_velocity.numpy().max())
    velocity = model.corrected_velocity.numpy().reshape(-1,3)
    
    u = velocity[:,0]
    v = velocity[:,1]
    z = velocity[:,2]
    inv_A = model.intermediate_velocity_step.inv_A.numpy().reshape(-1,3)

    results = m.pyvista_mesh
    # print(model.p.shape)
    results.cell_data['p'] = model.pressure_correction_step.p.numpy()
    results.cell_data['u'] = u
    results.cell_data['v'] = v
    results.cell_data['umag'] = np.sqrt(v**2 + u**2)
    results.cell_data['inv A x'] = inv_A[:,0]
    results.cell_data['inv A y'] = inv_A[:,1]
    results.cell_data['div u'] = b.div_u.numpy()
    results.cell_data['cell id'] = np.arange(model.num_cells)

    # results.plot(scalars="cell id", cmap="jet", show_scalar_bar=True)
    
    # 
    # # results.plot(scalars="inv A x", cmap="jet", show_scalar_bar=True)
    # # results.plot(scalars="inv A y", cmap="jet", show_scalar_bar=True)
    # results.plot(scalars="div u", cmap="jet", show_scalar_bar=True)

    centroids = model.struct_member_to_array('centroid','cells')
    x,y,z = [centroids[:,i].reshape(n,n) for i in range(3)]
    # print()
    v_05 = v.reshape((n,n))[n//2]
    u_05 = u.reshape((n,n))[:,n//2]
    model.Convergence.plot_residuals()
    results.plot(scalars="u", cmap="jet", show_scalar_bar=True)
    results.plot(scalars="p", cmap="jet", show_scalar_bar=True)
    results.plot(scalars="v", cmap="jet", show_scalar_bar=True)
    # results.plot(scalars="umag", cmap="jet", show_scalar_bar=True)
    
    # plt.plot(x[n//2],v_05)
    # plt.plot(x[n//2],inlet)
    # plt.show()

    
    y,P = np.sort( np.stack([m.cell_centroids[:,1],IC[:,-1]],axis = 0),axis = 1)

    # plt.plot(y,P)
    # plt.show()

    print(0.2/v_05.max())
    
    A = (1/n)*(1/n)
    V = m.cell_volumes[0]
    print(f'A {A} , 1/A {1/A}')
    print(f'V {V} , 1/V {1/V}')
    print(f'V/A {V/A} , A/V {A/V}')
