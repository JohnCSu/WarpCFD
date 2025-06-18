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

    def init_pressure_correction_operations(self):
        @wp.kernel
        def _calculate_divergence_kernel(cell_structs:wp.array(dtype=self.cell_struct),div_u:wp.array(dtype=self.float_dtype)):
            i,j = wp.tid() # C,F
            # calculate divergence
            wp.atomic_add(div_u,i,cell_structs[i].mass_fluxes[j])

        @wp.func
        def project_vector_to_face(face_idx:self.int_dtype,cell_struct:self.cell_struct,vec:Any):
            normal = cell_struct.face_normal[face_idx]
            return wp.dot(normal,wp.cw_mul(vec,normal))

        @wp.func
        def get_vector_from_array(arr:wp.array(ndim= 2,dtype=self.float_dtype),i:wp.int32):
            vec = wp.vector(arr[i][0],arr[i][1],arr[i][2],dtype= wp.float32)
            return vec


        @wp.kernel
        def _calculate_D_viscosity_kernel(D_face:wp.array(dtype=self.float_dtype),
                                          inv_A:wp.array2d(dtype=self.float_dtype),
                                          cell_structs: wp.array(dtype= self.cell_struct),
                                          face_structs:wp.array(dtype = self.face_struct)):
            face_id = wp.tid() # K
            # D is a C,3 array This is always 3
            if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
                cell_id = face_structs[face_id].adjacent_cells[0]
                face_idx = face_structs[face_id].cell_face_index[0]
                normal = cell_structs[cell_id].face_normal[face_idx]
                D_vec = get_vector_from_array(inv_A,cell_id)
                
                D_face[face_id] = wp.dot(normal,wp.cw_mul(normal,D_vec))
            else:
                owner_cell_id = face_structs[face_id].adjacent_cells[0]
                neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
                
                owner_D = get_vector_from_array(inv_A,owner_cell_id)
                neighbor_D =  get_vector_from_array(inv_A,neighbor_cell_id)

                owner_face_idx =  face_structs[face_id].cell_face_index[0]
                neighbor_face_idx =  face_structs[face_id].cell_face_index[1]
                owner_D_face = project_vector_to_face(owner_face_idx, cell_structs[owner_cell_id],owner_D)
                neighbor_D_face = project_vector_to_face(neighbor_face_idx, cell_structs[neighbor_cell_id],neighbor_D)

                D_face[face_id] = face_structs[face_id].norm_distance[0]*owner_D_face + face_structs[face_id].norm_distance[1]*neighbor_D_face
        @wp.kernel
        def _update_p_correction_rows_kernel(bsr_row_offsets:wp.array(dtype= self.int_dtype),
                                bsr_columns: wp.array(dtype= self.int_dtype),
                                values:wp.array(dtype=self.float_dtype),
                                b:wp.array(dtype=self.float_dtype),
                                fixed_cells_id:wp.array(dtype=self.int_dtype)):
            
            i = wp.tid() # We go through all Cells with fixed pressure values
            
            #For now only for pressure so cell_id = row
            cell_id = fixed_cells_id[i]
            col_start,col_end = bsr_row_offsets[cell_id], bsr_row_offsets[cell_id+1]
            for nnz_idx in range(col_start,col_end):
                column = bsr_columns[nnz_idx]
                if cell_id == column: #Diagonal
                    values[nnz_idx] = 1.
                else: # Off diagonal
                    values[nnz_idx] = 0.
            
            b[cell_id] = 0. # Replace the div u with 0 to indicate no pressure correction

        @wp.kernel
        def _interpolate_p_correction_to_faces_kernel(p_correction:wp.array(dtype= self.float_dtype),
                                               p_correction_face:wp.array(dtype=self.float_dtype),
                                               face_structs:wp.array(dtype=self.face_struct)):
            face_id = wp.tid()
            adjacent_cell_ids = face_structs[face_id].adjacent_cells
            owner_id = adjacent_cell_ids[0]

            if face_structs[face_id].is_boundary:
                p_correction_face[face_id] = p_correction[owner_id]                
            else:
                # Central Difference
                neighbor_id = adjacent_cell_ids[1]
                p_correction_face[face_id] = face_structs[face_id].norm_distance[0]*p_correction[owner_id] +face_structs[face_id].norm_distance[0]*p_correction[neighbor_id]


        @wp.kernel
        def _calculate_p_correction_gradient_kernel(p_correction_face:wp.array(dtype=self.float_dtype),
                                                    p_correction_gradient:wp.array(dtype=self.float_dtype),
                                                    inv_A:wp.array(dtype=self.float_dtype),
                                                    cell_structs:wp.array(dtype=self.cell_struct),
                                                    face_structs:wp.array(dtype=self.face_struct),
                                                    dimension:self.int_dtype):
            i,face_idx = wp.tid() # C,F
            row = dimension*i

            face_id = cell_structs[i].faces[face_idx]
            volume = cell_structs[i].volume
            normal = cell_structs[i].face_normal[face_idx]
            area = face_structs[face_id].area
            flux = p_correction_face[face_id]*normal*area/volume
            for dim in range(dimension):
                wp.atomic_add(p_correction_gradient,row + dim,flux[dim]*inv_A[row+dim])

        
        self._calculate_divergence = _calculate_divergence_kernel
        self._update_p_correction_rows = _update_p_correction_rows_kernel
        self._calculate_D_viscosity = _calculate_D_viscosity_kernel
        self._interpolate_p_correction_to_faces = _interpolate_p_correction_to_faces_kernel
        self._calculate_p_correction_gradient = _calculate_p_correction_gradient_kernel
    

    def calculate_divergence(self,arr:wp.array):
        arr.zero_()
        wp.launch(kernel=self._calculate_divergence, dim = (self.num_cells,self.faces_per_cell),inputs = [self.cells,arr])

    def calculate_p_correction_weights(self,D,inv_A):
        wp.launch(self._calculate_D_viscosity,dim = self.num_faces,inputs = [D,inv_A,self.cells,self.faces])
        self.weight_ops.calculate_laplacian(self.cells,self.faces,self.weights,self.p_index,viscosity=D)

    def update_p_correction_rows(self,bsr_matrix:sparse.BsrMatrix,div_u:wp.array):
        fixed_cells = self.cell_properties.fixed_cells
        wp.launch(kernel=self._update_p_correction_rows,dim = fixed_cells.shape[0], inputs=[bsr_matrix.offsets,bsr_matrix.columns,bsr_matrix.values,div_u,fixed_cells])

    def calculate_pressure_gradient(self,velocity_correction:wp.array,dimension = 3):
        '''
        Calculate u_correction = 1/a_p*grad_p_correction
        '''
        assert dimension*self.num_cells == velocity_correction.shape[0]
        velocity_correction.zero_()
        wp.launch(kernel=self._interpolate_p_correction_to_faces,dim = (self.num_faces),inputs = [self.p_correction,self.p_correction_face,self.faces])
        wp.launch(kernel=self._calculate_p_correction_gradient,dim = (self.num_cells,self.faces_per_cell),inputs = [self.p_correction_face,velocity_correction,self.inv_A,self.cells,self.faces,dimension])

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