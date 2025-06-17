import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp.types import vector
from src.preprocess import Mesh
import src.FV.Cells as Cells
from src.FV.Ops.mesh_ops import Mesh_Ops 
from warp import sparse
from warp.optim import linear
from typing import Any

import scipy.sparse as sp_sparse
from array_ops import add_1D_array,sub_1D_array,inv_1D_array,to_vector_array,mult_scalar_1D_array,absolute_1D_array,power_scalar_1D_array,sum_1D_array,L_norm_1D_array
wp.config.mode = "debug"

from matplotlib import pyplot as plt
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


class COO_Arrays:
    rows: wp.array
    cols: wp.array
    values: wp.array
    nnz: int
    offsets: wp.array

    def __init__(self,nnz_per_cell: wp.array,num_outputs,float_dtype,int_dtype):
        # Need to include cell itself

        nnz_per_cell:np.ndarray = nnz_per_cell.numpy()

        num_cells =nnz_per_cell.shape[0]
        nnz = int(num_outputs*nnz_per_cell.sum())
        offsets = np.zeros_like(nnz_per_cell)
        offsets[1:] = np.cumsum(num_outputs*nnz_per_cell)[:-1]        
        self.nnz = nnz

        self.offsets = wp.array(data = offsets)
        self.rows = wp.zeros(nnz,dtype=int_dtype)
        self.cols = wp.zeros(nnz,dtype=int_dtype)
        self.values = wp.zeros(nnz,dtype=float_dtype)



'''
    continuity_eps:float = 1e-4
    momentum_x_eps:float = 1e-4
    momentum_y_eps:float = 1e-4
    momentum_z_eps:float = 1e-4
    velocity_correction_eps:float = 1e-5
    pressure_correction_eps:float = 1e-5
'''


class Criteria:
    def __init__(self,name,eps = 1e-4,tol='relative',ord =1.,is_flux = False) -> None:
        assert tol == 'relative' or tol == 'abs'
        
        self.name = name
        self.eps = eps
        self.tol = tol
        self.ord = ord
        self.is_converged = False
        self.is_flux = is_flux
        self.non_zero = 1e-8
        if is_flux and (self.tol == 'abs'):
            raise ValueError('is_flux must be used with relative tolerance')

        
    def residual(self,arr:wp.array,rhs:wp.array|float = 0.,ord = None):
        if ord is None:
            ord = self.ord        
        res = np.linalg.norm((arr - rhs),ord)
        return res
    
    def relative_residual(self,arr:wp.array,rhs:wp.array|float,ord = None):
        ord = self.ord if ord is None else ord
        res = self.residual(arr,0.,ord) if self.is_flux else self.residual(arr,rhs,ord)
        return res/max(np.linalg.norm(rhs,ord),self.non_zero)
    
    def check(self,arr:wp.array|np.ndarray,rhs:wp.array|float|np.ndarray,ord =None):
        
        if isinstance(arr,wp.array):
            arr = arr.numpy()
        if isinstance(rhs,float):
            rhs = np.array([rhs],dtype = arr.dtype)
        elif isinstance(rhs,wp.array):
            rhs = rhs.numpy()
        
        if self.tol == 'abs':
            res =  self.residual(arr,rhs,ord)
        else:
            res =  self.relative_residual(arr,rhs,ord)

        if res < self.eps:
            self.is_converged = True    
        else:
            self.is_converged = False
        return res, self.is_converged

    
class Convergence:
    def __init__(self):
        self.continuity = Criteria('continuity',is_flux = True,eps = 1e-4)
        self.momentum_x = Criteria('momentum_x',ord =2.,eps = 1e-3)
        self.momentum_y = Criteria('momentum_y',ord =2.,eps = 1e-3)
        self.momentum_z = Criteria('momentum_z',ord =2.,eps = 1e-3)
        self.pressure_correction = Criteria('pressure_correction', tol = 'abs',eps = 1e-4,ord = np.inf )
        self.velocity_correction = Criteria('velocity_correction',tol = 'abs',eps = 1e-4 ,ord =  np.inf)
        self.criterias = [self.continuity,self.momentum_x,self.momentum_y,self.momentum_z,self.pressure_correction,self.velocity_correction]
        self.log_list = []
    def check(self,criteria:str,arr,rhs:float|wp.array = 0.,ord = None):
        criteria:Criteria = getattr(self,criteria)
        return criteria.check(arr,rhs,ord)
    
    def has_converged(self):
        for criteria in self.criterias:
            if criteria.is_converged == False:
                return False 
        return True
    def log(self,step:dict):
        self.log_list.append(step)


    def plot_residuals(self):
        x = np.arange(len(self.log_list))
        for criteria in self.criterias:
            name = criteria.name
            y = [step[name][0] for step in self.log_list]
            plt.plot(x,y, label = name)
        plt.ylim((0,1))
        plt.legend()
        plt.show()

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
        self.linear_solver = linear.bicgstab
        self.steps = 0
        self.MAX_STEPS = 2

        self.mesh_ops = Mesh_Ops(self.cell_struct,self.face_struct,self.node_struct,self.cell_properties,self.node_properties,self.num_outputs,self.float_dtype,self.int_dtype)
        
    def create_weight_struct(self,float_dtype = wp.float32):
        @wp.struct
        class Weights:
            convection_neighbor: float_dtype 
            convection_owner: float_dtype 

            laplacian_neighbor: float_dtype
            laplacian_owner: float_dtype
        return Weights


    def init_struct_operations(self):
        '''
        Initial Step of compiling functions for the following steps:
        - Apply BC
        - Face Interpolation 
        - Calculate Mass fluxes
        - Calculate Gradients
        - Calculate Convection And Laplacian Weights
        '''
        
        # @wp.kernel
        # def _apply_BC_kernel(face_struct:wp.array(dtype= self.face_struct),
        #                      boundary_condition:wp.array(dtype=self.face_properties.boundary_value.dtype),
        #                      gradient_condition:wp.array(dtype=self.face_properties.gradient_value.dtype),
        #                      boundary_ids: wp.array(dtype=self.face_properties.boundary_face_ids.dtype)):
            
        #     i,j = wp.tid() # B,num_outputs
            
        #     face_id = boundary_ids[i]

        #     if face_struct[face_id].value_is_fixed[j] == 1:
        #         face_struct[face_id].values[j] = boundary_condition[face_id][j]
        #         # wp.printf('%d %d %f %f \n',j,face_id,face_struct[face_id].values[j],boundary_condition[face_id][j])
        #     if face_struct[face_id].gradient_is_fixed[j] == 1:
        #         face_struct[face_id].gradients[j] = gradient_condition[face_id][j]


        # @wp.kernel
        # def _apply_cell_value_kernel(cell_struct:wp.array(dtype=self.cell_struct),
        #                              fixed_value: wp.array(dtype=self.cell_properties.fixed_value.dtype)):
        #     i,output = wp.tid() # C,O

        #     if cell_struct[i].value_is_fixed[output]:
        #         cell_struct[i].values[output] = fixed_value[i][output]
            

        # @wp.kernel
        # def _internal_calculate_face_interpolation_kernel(face_structs:wp.array(dtype=self.face_struct),
        #                                   cell_structs:wp.array(dtype = self.cell_struct),
        #                                   internal_face_ids:wp.array(dtype= self.face_properties.internal_face_ids.dtype),
        #                                   output_indices:wp.array(dtype=self.int_dtype),
        #                                   interpolation_method:int):
            
        #     '''
        #     We assume internal faces cannot have boundary conditions applied to them
        #     '''
        #     i,output_idx = wp.tid() # Loop through internal faces only
        #     output = output_indices[output_idx]
        #     face_id = internal_face_ids[i]
        #     adjacent_cell_ids = face_structs[face_id].adjacent_cells
        #     owner_cell = cell_structs[adjacent_cell_ids[0]]
        #     neighbor_cell = cell_structs[adjacent_cell_ids[1]] 

        #     if interpolation_method == 0: # Central Difference
        #         face_structs[face_id].values[output] = central_difference(owner_cell,neighbor_cell,face_structs[face_id],output)
        #     elif interpolation_method == 1: # Upwind
        #         face_structs[face_id].values[output] = upwind(owner_cell,neighbor_cell,face_structs[face_id],output)
        #     else:
        #         face_structs[face_id].values[output] = wp.nan
        # @wp.kernel
        # def _boundary_calculate_face_interpolation_kernel(face_structs:wp.array(dtype=self.face_struct),
        #                                   cell_structs:wp.array(dtype = self.cell_struct),
        #                                   boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
        #                                   output_indices:wp.array(dtype=self.int_dtype)):
            
        #     i,output_idx = wp.tid() # Loop through internal faces only
        #     output = output_indices[output_idx]
        #     face_id = boundary_face_ids[i]
        #     adjacent_cell_ids = face_structs[face_id].adjacent_cells
        #     owner_cell = cell_structs[adjacent_cell_ids[0]] # This is the only connected cell
        #     cell_face_idx = face_structs[face_id].cell_face_index[0]

        #     distance = wp.length(owner_cell.cell_centroid_to_face_centroid[cell_face_idx])

        #     if face_structs[face_id].gradient_is_fixed[output]: # We only do face interpolation if the gradient is fixed
        #         face_structs[face_id].values[output] = distance*face_structs[face_id].gradients[output] + owner_cell.values[output]

            

        # @wp.kernel
        # def _calculate_mass_flux_kernel(cell_structs:wp.array(dtype = self.cell_struct),
        #                           face_structs:wp.array(dtype = self.face_struct),
        # ):
            
        #     i,j = wp.tid() # C,F 
        #     face_id = cell_structs[i].faces[j]
        #     normal = cell_structs[i].face_normal[j]
        #     area = face_structs[face_id].area
            
        #     #Compute dot product
        #     dot_prod = wp.static(self.float_dtype(0.))
        #     for k in range(3):
        #       dot_prod += face_structs[face_id].values[k]*normal[k] 

        #     cell_structs[i].mass_fluxes[j] = dot_prod*area
            
        # @wp.kernel
        # def _calculate_gradients_kernel(cell_structs:wp.array(dtype = self.cell_struct),
        #                                 face_structs:wp.array(dtype = self.face_struct),
        #                                 node_structs:wp.array(dtype= self.node_struct),
        #                                 ):
            
        #     #Lets Use Gauss Linear from openFoam for now
        #     i = wp.tid() #C, faces_per_cell, num_outputs
            
        #     m = wp.matrix(shape= (wp.static(self.num_outputs),wp.static(self.dimension)),dtype= wp.float32)

        #     for face_idx in range(wp.static(self.faces_per_cell)):
        #         face_id = cell_structs[i].faces[face_idx]
        #         normal = cell_structs[i].face_normal[face_idx]
        #         m += wp.outer(face_structs[face_id].values,normal)*face_structs[face_id].area/cell_structs[i].volume
        #     cell_structs[i].gradients = m
        @wp.kernel
        def _calculate_convection_weights_kernel(cell_structs:wp.array(dtype = self.cell_struct),
                                face_structs:wp.array(dtype = self.face_struct),
                                weights:wp.array(ndim=3,dtype=self.weights.dtype),
                                outputs:wp.array(dtype=self.int_dtype),
                                interpolation:int
                                ):
            
            i,j,k = wp.tid() # loop through C,F,O
            output = outputs[k]
            face_idx = cell_structs[i].faces[j]
            
            if face_structs[face_idx].is_boundary == 1:
                weights[i,j,output].convection_owner = wp.static(self.float_dtype(0.)) # Set the contribtuion to owner to 0 as for boundary term goes to RHS
                weights[i,j,output].convection_neighbor = cell_structs[i].mass_fluxes[j]*face_structs[face_idx].values[output]
            else:
                
                owner_face_idx = cell_structs[i].face_sides[j] # Returns if current cell i is on the 0 side of face or 1 side of face
                adj_face_idx = wp.static(self.int_dtype(1)) - owner_face_idx # Apply not operation to get the other index (can only be 1 or 0)
                if interpolation == 0: # Central Differencing
                    weights[i,j,output].convection_owner = face_structs[face_idx].norm_distance[owner_face_idx]*cell_structs[i].mass_fluxes[j]
                    weights[i,j,output].convection_neighbor = face_structs[face_idx].norm_distance[adj_face_idx]*cell_structs[i].mass_fluxes[j]
                elif interpolation == 1: # Upwind
                    if cell_structs[i].mass_fluxes[j] > 0:
                        weights[i,j,output].convection_owner = cell_structs[i].mass_fluxes[j]
                        weights[i,j,output].convection_neighbor = 0.

                    else:
                        weights[i,j,output].convection_owner = 0.
                        weights[i,j,output].convection_neighbor = -cell_structs[i].mass_fluxes[j]
        @wp.kernel
        def _interpolate_viscosity_to_face_kernel(cell_viscosity:wp.array(dtype=self.float_dtype),face_viscosity:wp.array(dtype=self.float_dtype),face_structs:wp.array(dtype=self.face_struct)):
            face_id = wp.tid()
            owner_cell_id = face_structs[face_id].adjacent_cells[0]
            neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
            if face_structs[face_id].is_boundary:
                face_viscosity[face_id] = cell_viscosity[owner_cell_id]
            else:
                face_viscosity[face_id] = cell_viscosity[owner_cell_id]*face_structs[face_id].norm_distance[0] + cell_viscosity[neighbor_cell_id]*face_structs[face_id].norm_distance[1]


        @wp.kernel
        def _calculate_laplacian_weights_kernel(cell_structs:wp.array(dtype = self.cell_struct),
                                face_structs:wp.array(dtype = self.face_struct),
                                weights:wp.array3d(dtype=self.weights.dtype),
                                viscosity:wp.array(dtype=self.float_dtype),
                                output_indices:wp.array(dtype=self.int_dtype)):
            i,j,output_idx = wp.tid() # C,F,O
            face_id = cell_structs[i].faces[j]
            output = output_indices[output_idx]
            
            if viscosity.shape[0] == 1: # if viscosity is constant
                nu = viscosity[0]
            else:
                nu = viscosity[face_id] 

            # we calculate grad(u)_f dot n_f with central differencing
            if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
                if face_structs[face_id].gradient_is_fixed[output]:
                    weights[i,j,output].laplacian_owner = 0.
                    weights[i,j,output].laplacian_neighbor = face_structs[face_id].gradients[output]
                else:
                    cell_centroid_to_face_centroid_magnitude = wp.length(cell_structs[i].cell_centroid_to_face_centroid[j])
                    weight = nu*(face_structs[face_id].area)/cell_centroid_to_face_centroid_magnitude

                    weights[i,j,output].laplacian_owner = -weight # Unknown 
                    weights[i,j,output].laplacian_neighbor = weight*(face_structs[face_id].values[output]) # Known Value so evaluate explicit
                # wp.printf('scalar %f nu %f area %f dist %f val %f  \n ',vis_area_div_dist,nu,face_structs[face_id].area,cell_centroid_to_face_centroid_magnitude,face_structs[face_id].values[output])

                # wp.printf('a %f fv %f cv %f \n',face_structs[face_id].area,face_structs[face_id].values[output],cell_structs[i].values[output])
            else:
                distance = wp.length(face_structs[face_id].cell_distance)
                weight = nu*(face_structs[face_id].area)/distance
                weights[i,j,output].laplacian_owner = -weight
                weights[i,j,output].laplacian_neighbor = weight
                # wp.printf('%f neighbor \n',weights[i,j,output].laplacian_neighbor)

        
        # self._apply_BC = _apply_BC_kernel
        # self._apply_cell_value = _apply_cell_value_kernel
        # self._internal_calculate_face_interpolation = _internal_calculate_face_interpolation_kernel
        # self._boundary_calculate_face_interpolation = _boundary_calculate_face_interpolation_kernel
        # self._calculate_mass_flux = _calculate_mass_flux_kernel 
        # self._calculate_gradients = _calculate_gradients_kernel
        self._calculate_convection_weights = _calculate_convection_weights_kernel
        self._calculate_laplacian_weights = _calculate_laplacian_weights_kernel
        self._interpolate_viscosity_to_face = _interpolate_viscosity_to_face_kernel
    def init_matrix_operations(self):
        
        @wp.kernel
        def get_matrix_indices( rows:wp.array(dtype=self.int_dtype),
                                cols:wp.array(dtype=self.int_dtype),
                                offsets:wp.array(dtype=self.int_dtype),
                                cell_structs: wp.array(dtype=self.cell_struct),
                                face_structs:wp.array(dtype=self.face_struct),
                                num_outputs: self.int_dtype):
            i,nnz_cell_offset,output = wp.tid() # Loop C,NNZ,D (set to 3)
            D = num_outputs
            cell_offset = offsets[i]
            num_nnz_per_cell = cell_structs[i].nnz
            # wp.printf('i:%d offset:%d output: %d \n',i,nnz_cell_offset,output)   
            if nnz_cell_offset == 0: # We are weighing the diagonal
                total_offset = cell_offset + (num_nnz_per_cell)*output
                rows[total_offset] = D*i + output
                #Because of structure, if 0 then we just record that the diagonal is needed. Updates to values is made when we iterate over face_idx
                cols[total_offset] = rows[total_offset]
                # wp.printf('i:%d offset:%d num_nnz_per_cell %d nnz_cell_offset:%d output:%d r:%d c:%d t:%d \n', i,cell_offset,num_nnz_per_cell,nnz_cell_offset,output,rows[total_offset],cols[total_offset],total_offset)  
            else:
                # Find the neighbor cell id corresponding to the face idx which gives us the column
                face_idx = nnz_cell_offset - 1 # 0 is considered the diagonal (self)
                face_id = cell_structs[i].faces[face_idx]
                
                if face_structs[face_id].is_boundary == 0: # Internal Face
                    face_offset = cell_structs[i].face_offset_index[face_idx]
                    total_offset = cell_offset + (num_nnz_per_cell)*output + face_offset # Need Face offset so for each face index from 1->(F+1)
                    rows[total_offset] = D*i + output
                    neighbor_cell_id = cell_structs[i].neighbors[face_idx]
                    cols[total_offset] = D*neighbor_cell_id + output
                    # wp.printf('i:%d offset:%d num_nnz_per_cell %d nnz_cell_offset:%d output:%d r:%d c:%d t:%d \n', i,cell_offset,num_nnz_per_cell,nnz_cell_offset,output,rows[total_offset],cols[total_offset],total_offset)  
            # wp.printf('i:%d offset:%d num_nnz_per_cell %d output:%d \n', i,cell_offset,nnz_cell_offset,output)
        

        @wp.func
        def get_face_idx(vec1:Any,id:int):
            for i in range(vec1.length):
                if vec1[i] == id:
                    return i
            
            return -1
        
        @wp.kernel
        def _calculate_BSR_values_kernel(bsr_rows:wp.array(dtype=int),
                                bsr_columns:wp.array(dtype=int),
                                values:wp.array(dtype= self.float_dtype),
                                cell_structs:wp.array(dtype= self.cell_struct),
                                weights:wp.array3d(dtype= self.weight_struct),
                                output_indices:wp.array(dtype= self.int_dtype)):
            i = wp.tid()

            row = bsr_rows[i]
            col = bsr_columns[i]
            num_outputs = output_indices.shape[0]
            output_idx = wp.mod(row,num_outputs)
            output = output_indices[output_idx]
            cell_id = row//num_outputs
            neighbor_id = col//num_outputs

            owner_cell = cell_structs[cell_id]

            if row == col:
                for face_idx in range(owner_cell.faces.length):
                    # if owner_cell.neighbors[face_idx] != -1: # So not boundary
                    wp.atomic_add(values,i, -weights[cell_id,face_idx,output].laplacian_owner + weights[cell_id,face_idx,output].convection_owner)
            else: # Off Diagonal
                face_idx = get_face_idx(owner_cell.neighbors,neighbor_id)
                if face_idx != -1:    
                    wp.atomic_add(values,i, -weights[cell_id,face_idx,output].laplacian_neighbor + weights[cell_id,face_idx,output].convection_neighbor)

        @wp.kernel
        def _calculate_RHS_values_kernel(b:wp.array(dtype=self.float_dtype),
                                        boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
                                        face_structs:wp.array(dtype=self.face_struct),
                                        weights:wp.array3d(dtype= self.weight_struct),
                                        output_indices:wp.array(dtype= self.int_dtype)):
            
            i,output_idx = wp.tid() #Loop through Boundary faces

            face_id = boundary_face_ids[i]
            cell_id = face_structs[face_id].adjacent_cells[0]
            face_idx = face_structs[face_id].cell_face_index[0]
            output = output_indices[output_idx]
            row = cell_id*output_indices.shape[0] +output_idx

            wp.atomic_add(b,row,weights[cell_id,face_idx,output].laplacian_neighbor -  weights[cell_id,face_idx,output].convection_neighbor )


        @wp.kernel
        def form_p_grad_vector_kernel(b:wp.array(dtype=self.float_dtype),
                                            cell_structs: wp.array(dtype=self.cell_struct),density:self.float_dtype):
                i,dim = wp.tid() # Go by 
                D = wp.static(self.dimension)
                p = wp.static(self.int_dtype(3))
                row = i*D + dim
                # wp.printf('n1 %f n2 %f n3 %f \n',cell_structs[i].grads[p][0],cell_structs[i].grads[p][1],cell_structs[i].grads[p][2])
                b[row] =  cell_structs[i].gradients[p][dim]/density


        @wp.kernel
        def fill_x_array_from_struct_kernel(x:wp.array(dtype=self.float_dtype),cell_structs:wp.array(dtype=self.cell_struct),output_indices:wp.array(dtype=self.int_dtype)):
            i,output_idx = wp.tid() # C,O
            
            output = output_indices[output_idx]
            num_outputs = output_indices.shape[0]
            row = i*num_outputs + output_idx
            x[row] = cell_structs[i].values[output]

        @wp.kernel
        def fill_struct_from_x_array_kernel(x:wp.array(dtype=self.float_dtype),cell_structs:wp.array(dtype=self.cell_struct),output_indices:wp.array(dtype=self.int_dtype)):

            i,output_idx = wp.tid() # C,O
            
            output = output_indices[output_idx]
            num_outputs = output_indices.shape[0]
            row = i*num_outputs + output_idx
            cell_structs[i].values[output] = x[row] 
            
        @wp.kernel
        def _massflux_to_array_kernel(cell_structs:wp.array(dtype=self.cells.dtype),out:wp.array2d(dtype = self.float_dtype)):
            i,j = wp.tid()
            out[i,j] = cell_structs[i].mass_fluxes[j]


        self._form_p_grad_vector_kernel= form_p_grad_vector_kernel
        '''attribute to call method with which to form the pressure gradient vector'''
        self._massflux_to_array = _massflux_to_array_kernel
        # self._calculate_BSR_matrix_and_RHS = calculate_BSR_matrix_and_RHS_kernel
        self._calculate_BSR_values = _calculate_BSR_values_kernel
        self._calculate_RHS_values = _calculate_RHS_values_kernel

        self._get_matrix_indices = get_matrix_indices

        self._fill_x_array_from_struct=fill_x_array_from_struct_kernel
        self._fill_struct_from_x_array = fill_struct_from_x_array_kernel


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
    

    def calculate_convection(self,output_indices:wp.array,interpolation_method = 1):
        num_outputs = len(output_indices)
        wp.launch(kernel=self._calculate_convection_weights,dim = (self.num_cells,self.faces_per_cell,num_outputs),inputs = [self.cells,self.faces,self.weights,output_indices,interpolation_method])
    

    def calculate_laplacian(self,output_indices:wp.array,viscosity: float | wp.array):
        num_outputs = len(output_indices)
        if isinstance(viscosity,float):
            viscosity = wp.array([viscosity],dtype =self.float_dtype)

        elif isinstance(viscosity,wp.array):
            assert len(viscosity.shape) == 1
            if viscosity.shape[0] == self.num_cells:
                if self.face_viscosity is None:
                    self.face_viscosity = wp.zeros(shape = self.num_faces,dtype= self.float_dtype)
                wp.launch(self._interpolate_viscosity_to_face,dim = self.num_faces, inputs = [viscosity,self.face_viscosity,self.faces])
            
                viscosity = self.face_viscosity
            else:
                assert viscosity.shape[0] == self.num_faces
        else:
            raise ValueError('viscosity can be type float or wp.array')

        wp.launch(kernel=self._calculate_laplacian_weights,dim = (self.num_cells,self.faces_per_cell,num_outputs),inputs = [self.cells,self.faces,self.weights,viscosity,output_indices])


    def calculate_BSR_matrix_indices(self,COO_array:COO_Arrays,num_outputs:int):
        wp.launch(kernel=self._get_matrix_indices,dim = [self.num_cells,self.faces_per_cell+1,num_outputs],inputs = [COO_array.rows,
                                                                                                                  COO_array.cols,
                                                                                                                  COO_array.offsets,
                                                                                                                  self.cells,
                                                                                                                  self.faces,
                                                                                                                  num_outputs
                                                                                                                  ])


    def calculate_BSR_matrix_and_RHS(self,BSR_matrix:sparse.BsrMatrix,output_indices:wp.array,b:wp.array = None,rows =None):
        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        assert rows.shape[0] == BSR_matrix.values.shape[0]
        BSR_matrix.values.zero_() # Reset to zero
        wp.launch(kernel=self._calculate_BSR_values,dim = BSR_matrix.values.shape[0],inputs=[rows,
                                                                                             BSR_matrix.columns,
                                                                                             BSR_matrix.values,
                                                                                             self.cells,
                                                                                             self.weights,
                                                                                             output_indices])
        
        if b is not None:
            b.zero_()
            boundary_ids = self.face_properties.boundary_face_ids
            wp.launch(kernel= self._calculate_RHS_values, dim = (boundary_ids.shape[0],output_indices.shape[0]),inputs = [b,
                                                                                                                    boundary_ids,
                                                                                                                    self.faces,
                                                                                                                    self.weights,
                                                                                                                    output_indices,

            ])
    def form_p_grad_vector(self):
        wp.launch(kernel=self._form_p_grad_vector_kernel, dim = [self.num_cells,self.dimension], inputs = [self.grad_P,self.cells,self.density])
        
    def fill_x_array_from_struct(self,x,output_indices):
        
        assert x.shape[0] == self.cells.shape[0]*output_indices.shape[0]
        wp.launch(kernel=self._fill_x_array_from_struct,dim = [self.num_cells,output_indices.shape[0]],inputs = [x,self.cells,output_indices])
    
    def fill_struct_from_x_array(self,x,output_indices):
        assert x.shape[0] == self.cells.shape[0]*output_indices.shape[0]
        wp.launch(kernel=self._fill_struct_from_x_array,dim = [self.num_cells,output_indices.shape[0]],inputs = [x,self.cells,output_indices])

    def get_b_vector(self):
        sub_1D_array(self.H,self.grad_P,self.b)
    
    def solve_Axb(self,A:sparse.BsrMatrix,x:wp.array,b:wp.array):
        M = linear.preconditioner(A)
        # print(b.numpy())
        return self.linear_solver(A =A,M= M, b = b,x = x,maxiter=500)

    def massflux_to_array(self,arr:wp.array):
        assert len(arr.shape) == 2 and arr.shape[0] == self.num_cells and arr.shape[1] == self.faces_per_cell
        wp.launch(kernel=self._massflux_to_array,dim = [self.num_cells,self.faces_per_cell],inputs = [self.cells,arr])


    def calculate_divergence(self,arr:wp.array):
        arr.zero_()
        wp.launch(kernel=self._calculate_divergence, dim = (self.num_cells,self.faces_per_cell),inputs = [self.cells,arr])

    def calculate_p_correction_weights(self,D,inv_A):
        wp.launch(self._calculate_D_viscosity,dim = self.num_faces,inputs = [D,inv_A,self.cells,self.faces])
        self.calculate_laplacian(self.p_index,viscosity=D)

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


    def init_intermediate_velocity_step(self):
        self.massflux_array = wp.empty(shape = (self.num_cells,self.faces_per_cell))
        self.vel_COO_array = COO_Arrays(self.cell_properties.nnz_per_cell,self.dimension,self.float_dtype,self.int_dtype)
        '''Arrays for Sparse Matrix for U step'''
        # WE mak A*u = B matrix A. In 3D we have 3
        self.sparse_dimensions = (self.num_cells*self.dimension,self.num_cells*self.dimension)
        self.H = wp.zeros(shape=(self.dimension*self.num_cells),dtype=self.float_dtype)
        self.grad_P = wp.zeros_like(src = self.H)
        self.b = wp.zeros_like(src = self.H)
        self.intermediate_vel = wp.zeros_like(src = self.H)
        self.initial_vel = wp.zeros_like(src = self.H)

        self.calculate_BSR_matrix_indices(self.vel_COO_array,num_outputs = 3)
        
        '''
        Each Cell results in 3 rows. Each Row would have 1 + num neighbors. We can actually calculate this beforehand
        That means each cell gives 3*(1+num_neighbors) of nnz. So total nnz = 3*(1+num_neighbors)*num_cells 
        '''
        
        self.vel_matrix = sparse.bsr_from_triplets(*self.sparse_dimensions,
                                                      rows = self.vel_COO_array.rows,
                                                      columns = self.vel_COO_array.cols,
                                                      values = self.vel_COO_array.values,
                                                      prune_numerical_zeros= False)
        
        self.vel_matrix_rows = self.vel_matrix.uncompress_rows()

    def init_pressure_correction_step(self):
        self.p = wp.zeros(shape= self.num_cells,dtype=self.float_dtype)
        self.p_relaxation_factor = 0.02
        self.u_relaxation_factor = 0.5
        self.p_COO_array = COO_Arrays(self.cell_properties.nnz_per_cell,1,self.float_dtype,self.int_dtype)
        self.inv_A = wp.zeros(shape = self.vel_matrix.shape[0],dtype = self.float_dtype)
        self.div_u = wp.zeros(shape = self.num_cells,dtype = self.float_dtype)
        self.D_face = wp.zeros(shape= self.num_faces, dtype= self.float_dtype)
        self.p_correction = wp.zeros(shape=self.num_cells, dtype= self.float_dtype)
        self.p_correction_face = wp.zeros(shape= self.num_faces, dtype= self.float_dtype)
        self.velocity_correction = wp.zeros_like(self.inv_A)

        '''Arrays for Sparse Matrix fro pressure correction'''
        self.calculate_BSR_matrix_indices(self.p_COO_array,1)
        self.p_correction_matrix = sparse.bsr_from_triplets(self.num_cells,self.num_cells,
                                                            rows = self.p_COO_array.rows,
                                                            columns = self.p_COO_array.cols,
                                                            values = self.p_COO_array.values,
                                                            prune_numerical_zeros= False)
        self.p_correction_matrix_rows = self.p_correction_matrix.uncompress_rows()

    def solve_intermediate_velocity(self):
        
        self.fill_x_array_from_struct(self.intermediate_vel,self.vel_indices)
        self.form_p_grad_vector()
        self.get_b_vector()
        # print(self.grad_P)
        wp.copy(self.intermediate_vel,self.initial_vel) # Use velocity at current iteration as initial guess
        result = self.solve_Axb(A = self.vel_matrix,x = self.intermediate_vel,b = self.b)
        print(result)
        self.fill_struct_from_x_array(self.intermediate_vel,self.vel_indices)
        
    def update_flux_and_weights(self):
        self.calculate_face_interpolation(self.vel_indices,interpolation_method=1)
        self.calculate_face_interpolation(self.p_index,interpolation_method=0)
        self.calculate_mass_flux()
        self.calculate_gradients()
        self.calculate_convection(output_indices= self.vel_indices)
        self.calculate_laplacian(output_indices= self.vel_indices,viscosity=self.viscosity/self.density)
        self.calculate_BSR_matrix_and_RHS(self.vel_matrix,output_indices=self.vel_indices,b = self.H,rows= self.vel_matrix_rows)
        print('diag',sparse.bsr_get_diag(self.vel_matrix).numpy().mean())
        self.inv_A = sparse.bsr_get_diag(self.vel_matrix)
        inv_1D_array(self.inv_A,self.inv_A) # Invert A i.e. 1/a_i


    def solve_pressure_correction(self):


        self.calculate_face_interpolation(self.vel_indices,interpolation_method=1)
        self.calculate_face_interpolation(self.p_index,interpolation_method=0)
        self.calculate_mass_flux() # Update Mass Flux
        self.calculate_divergence(self.div_u)
        
        
        
        

        self.calculate_p_correction_weights(self.D_face,self.inv_A.reshape((-1,3)))
        # print(self.D_face)
        self.calculate_BSR_matrix_and_RHS(self.p_correction_matrix,output_indices=self.p_index,rows = self.p_correction_matrix_rows) # only calculate BSR Values
        mult_scalar_1D_array(self.p_correction_matrix.values,-1.,self.p_correction_matrix.values) # the diag is -ve so
        
        self.update_p_correction_rows(self.p_correction_matrix,self.div_u)
        self.solve_Axb(self.p_correction_matrix,self.p_correction,self.div_u)
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
        # print('invA',np.abs(self.inv_A.numpy()).max(),np.abs(self.vel_matrix.values.numpy()).min())
        # print('p max: ',np.abs(self.p_correction.numpy().max()))
        # print('p min: ',np.abs(self.p_correction.numpy().min()))
        # print('initial vel',np.abs(self.initial_vel.numpy()).max())
        # print('correction velocity',np.abs(self.velocity_correction.numpy()).max())
        self.fill_struct_from_x_array(self.initial_vel,self.vel_indices)
        # Update pressure
        self.fill_x_array_from_struct(self.p,self.p_index)
        mult_scalar_1D_array(self.p_correction,self.p_relaxation_factor,self.p_correction)
        add_1D_array(self.p,self.p_correction,self.p)
        self.fill_struct_from_x_array(self.p,self.p_index)

    def check_convergence(self,vel_matrix:sparse.BsrMatrix,velocity:wp.array,b:wp.array,div_u:wp.array,velocity_correction:wp.array,p_correction:wp.array):
        '''
        Checks the following:
        divergence residual
        momentum equations
        pressure correction residual
        '''
        
        self.massflux_to_array(self.massflux_array)
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
        
        self.init_struct_operations()
        self.init_matrix_operations()
        self.init_pressure_correction_operations()

        self.init_intermediate_velocity_step()
        self.init_pressure_correction_step()

        self.apply_BC()
        self.apply_cell_value()

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
            
        self.solve_intermediate_velocity()
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