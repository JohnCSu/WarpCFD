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


from warp_cfd.preprocess import Mesh
import warp_cfd.FV.Cells as Cells
from warp_cfd.FV.Ops import Mesh_Ops 
from warp_cfd.FV.Ops import Weights_Ops
from warp_cfd.FV.Ops import Matrix_Ops
from warp_cfd.FV.Convergence import Convergence
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.intermediate_velocity import intermediate_velocity_step

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

        #Field Outputs
        self.field_output_vector = vector(length = len(self.output_variables),dtype= self.float_dtype)
        self.field_output = wp.zeros(shape = (self.num_cells),dtype=self.field_output_vector)
        self.field_output_gradient_matrix = wp.mat(shape = (len(self.output_variables),3),dtype= self.float_dtype)
        self.field_output_gradients = wp.zeros(shape = self.num_cells,dtype=self.field_output_gradient_matrix)
        
        self.cell_struct,self.face_struct,self.node_struct= Cells.create_mesh_structs(self.nodes_per_cell,self.faces_per_cell,self.nodes_per_face,self.num_outputs,self.dimension,self.float_dtype,self.int_dtype)
        self.weight_struct = self.create_weight_struct(self.float_dtype)
        self.cells = wp.zeros(shape=self.num_cells,dtype= self.cell_struct)
        '''Array of cell structs for storing information about cell dependent values'''
        self.faces = wp.zeros(shape = self.num_faces,dtype=self.face_struct)
        '''Array of face structs for storing information about face dependent values'''
        self.nodes = wp.zeros(shape= self.num_nodes,dtype=self.node_struct)
        '''Array of node structs for storing inforation about nodes (mainly id and coordinates)'''
        self.weights = wp.zeros(shape = (self.num_cells,self.faces_per_cell,self.num_outputs),dtype= self.weight_struct)
        '''Array of structs of size (C,F,O) Store Cells Struct which contains cell info'''
        
        self.face_viscosity = None
        
        self.steps = 0
        self.MAX_STEPS = 2

        Ops_args = [self.cell_struct,self.face_struct,self.node_struct,self.weight_struct,self.cell_properties,self.face_properties,self.num_outputs,self.float_dtype,self.int_dtype]

        self.mesh_ops = Mesh_Ops(*Ops_args)
        self.weight_ops = Weights_Ops(*Ops_args) 
        self.matrix_ops = Matrix_Ops(*Ops_args)

    def create_weight_struct(self,float_dtype = wp.float32):
        @wp.struct
        class Weights:
            convection_neighbor: float_dtype 
            convection_owner: float_dtype 

            laplacian_neighbor: float_dtype
            laplacian_owner: float_dtype
        return Weights


    def calculate_p_correction_weights(self,D,inv_A,cells,faces,weights,output_indices):
        self.calculate_D_viscosity(D,inv_A,cells,faces)
        self.weight_ops.calculate_laplacian(cells,faces,weights,output_indices,viscosity=D)


    def init_pressure_correction_step(self):
        self.p = wp.zeros(shape= self.num_cells,dtype=self.float_dtype)
        self.p_relaxation_factor = 0.02
        self.u_relaxation_factor = 0.5
        self.p_COO_array = COO_Arrays(self.cell_properties.nnz_per_cell,1,self.float_dtype,self.int_dtype)
        self.div_u = wp.zeros(shape = self.num_cells,dtype = self.float_dtype)
        self.D_face = wp.zeros(shape= self.num_faces, dtype= self.float_dtype)
        self.p_correction = wp.zeros(shape=self.num_cells, dtype= self.float_dtype)
        self.p_correction_face = wp.zeros(shape= self.num_faces, dtype= self.float_dtype)
        self.velocity_correction = wp.zeros(shape = self.num_cells*self.dimension, dtype= self.float_dtype)

        '''Arrays for Sparse Matrix fro pressure correction'''
        self.matrix_ops.calculate_BSR_matrix_indices(self.p_COO_array,self.cells,self.faces,1)
        self.p_correction_matrix = sparse.bsr_from_triplets(self.num_cells,self.num_cells,
                                                            rows = self.p_COO_array.rows,
                                                            columns = self.p_COO_array.cols,
                                                            values = self.p_COO_array.values,
                                                            prune_numerical_zeros= False)
        self.p_correction_matrix_rows = self.p_correction_matrix.uncompress_rows()

        
    def update_flux_and_weights(self):
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.vel_indices,interpolation_method=1)
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.p_index,interpolation_method=0)
        self.mesh_ops.calculate_mass_flux(self.cells,self.faces) # Update Mass Flux
        self.mesh_ops.calculate_gradients(self.cells,self.faces,self.nodes)

        self.weight_ops.calculate_convection(self.cells,self.faces,self.weights,output_indices= self.vel_indices)
        self.weight_ops.calculate_laplacian(self.cells,self.faces,self.weights,output_indices= self.vel_indices,viscosity=self.viscosity/self.density)
        


    def solve_pressure_correction(self):
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.vel_indices,interpolation_method=1)
        self.mesh_ops.calculate_face_interpolation(self.cells,self.faces,self.p_index,interpolation_method=0)
        self.mesh_ops.calculate_mass_flux(self.cells,self.faces) # Update Mass Flux
        self.calculate_divergence(self.div_u)

        self.calculate_p_correction_weights(self.D_face,self.inv_A.reshape((-1,3)))
        # print(self.D_face)
        self.matrix_ops.calculate_BSR_matrix_and_RHS(self.p_correction_matrix,self.cells,self.faces,self.weights,output_indices=self.p_index,rows = self.p_correction_matrix_rows) # only calculate BSR Values
        mult_scalar_1D_array(self.p_correction_matrix.values,-1.,self.p_correction_matrix.values) # the diag is -ve so
        
        self.update_p_correction_rows(self.p_correction_matrix,self.div_u)
        self.matrix_ops.solve_Axb(self.p_correction_matrix,self.p_correction,self.div_u)
        self.calculate_pressure_gradient(self.velocity_correction)

        # Calc velocity correction - 1/ap *gradp'
        mult_scalar_1D_array(self.velocity_correction,-1.,self.velocity_correction)
        # Create Delta u_new := uinter - u_correcction
        add_1D_array(self.intermediate_vel,self.velocity_correction,self.velocity_correction)
        # Relax the velocity
        mult_scalar_1D_array(self.velocity_correction,self.u_relaxation_factor,self.velocity_correction)
        mult_scalar_1D_array(self.initial_vel,1-self.u_relaxation_factor,self.initial_vel)
        add_1D_array(self.initial_vel,self.velocity_correction,self.initial_vel)

        print('pgrad',np.abs(self.velocity_correction.numpy().reshape(-1,3)).max())
        self.matrix_ops.fill_struct_from_x_array(self.initial_vel,self.cells,self.vel_indices)
        # Update pressure
        self.matrix_ops.fill_x_array_from_struct(self.p,self.cells,self.p_index)
        mult_scalar_1D_array(self.p_correction,self.p_relaxation_factor,self.p_correction)
        add_1D_array(self.p,self.p_correction,self.p)
        self.matrix_ops.fill_struct_from_x_array(self.p,self.cells,self.p_index)

    def check_convergence(self,vel_matrix:sparse.BsrMatrix,velocity:wp.array,b:wp.array,div_u:wp.array,velocity_correction:wp.array,p_correction:wp.array):
        '''
        Checks the following:
        divergence residual
        momentum equations
        pressure correction residual
        '''
        
        self.matrix_ops.massflux_to_array(self.massflux_array,self.cells)
        div_u = wp.zeros_like(div_u)
        self.calculate_divergence(div_u)
        Ax:wp.array = vel_matrix @ velocity

        Ax = Ax.numpy().reshape(-1,3)
        b = b.numpy().reshape(-1,3)

        # print(Ax[:,1],b[:,1])
        convergence = {
            'momentum_x':self.Convergence.check('momentum_x',Ax[:,0],b[:,0]),
            'momentum_y':self.Convergence.check('momentum_y',Ax[:,1],b[:,1]),
            'momentum_z':self.Convergence.check('momentum_z',Ax[:,2],b[:,2]),
            'continuity':self.Convergence.check('continuity',div_u,self.massflux_array.flatten()),
            'velocity_correction':self.Convergence.check('velocity_correction',velocity_correction),
          'pressure_correction':self.Convergence.check('pressure_correction',p_correction),
        }
        
        print(convergence)
        self.Convergence.log(convergence)
        return self.Convergence.has_converged()
    def init_step(self):
        '''
        Initialises The Model. For now assume that the mesh and BC are both Fixed in Time
        '''
        Cells.init_structs(self.cells,self.faces,self.nodes,self.cell_properties,self.face_properties,self.node_properties)
        self.mesh_ops.init()
        self.weight_ops.init()
        self.matrix_ops.init()
        self.intermediate_velocity_step = intermediate_velocity_step(self.mesh_ops,self.weight_ops,self.matrix_ops)
        self.intermediate_velocity_step.init(self.cells,self.faces)
        self.init_pressure_correction_operations()
        self.init_pressure_correction_step()

        self.mesh_ops.apply_BC(self.faces)
        self.mesh_ops.apply_cell_value(self.cells)


        self.massflux_array = wp.empty(shape = (self.num_cells,self.faces_per_cell))
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
            self.update_flux_and_weights()

        self.vel_matrix,self.initial_vel,self.b,self.intermediate_vel,self.inv_A = self.intermediate_velocity_step.solve(self.cells,self.faces,self.weights,self.density,self.vel_indices)

        self.solve_pressure_correction()
        self.update_flux_and_weights()
        converged = self.check_convergence(self.vel_matrix,self.initial_vel,self.b,self.div_u,self.velocity_correction,self.p_correction)
        self.steps += 1
        return converged

        

def bsr_to_coo_array(bsr:sparse.BsrMatrix):
    return sp_sparse.coo_array((bsr.values.numpy(),(bsr.uncompress_rows().numpy(),bsr.columns.numpy())),shape = (bsr.nrow,bsr.ncol))

if __name__ == '__main__':
    from grid import create_hex_grid
    n = 3
    m = create_hex_grid(n,n,1,(0.1/n,0.1/n,0.01))
    m.set_boundary_value('+X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-Y',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('+Y',u = 1,v = 0,w = 0) # Velocity Inlet
    
    m.set_gradient_value('-Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+Z',u=0,v=0,w=0,p=0) # No penetration condition

    m.set_cell_value(0,p= 0)

    # print(m.nodes)
    
    model = FVM(m,density = 1,viscosity= 0.01)
    model.init_step()
    np.set_printoptions(linewidth=300,threshold=100000,precision = 5)
    model.MAX_STEPS = 2
    for i in range(model.MAX_STEPS):
        print(model.step())
        
    model.Convergence.plot_residuals()
    
    velocity = model.initial_vel.numpy().reshape(-1,3)
    u = velocity[:,0]
    results = m.pyvista_mesh
    print(model.p.shape)
    results.cell_data['p'] = model.p.numpy()
    results.cell_data['u'] = u
    results.plot(scalars="p", cmap="jet", show_scalar_bar=True)