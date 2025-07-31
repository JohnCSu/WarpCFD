import warp as wp
from typing import Any
from .ops_class import Ops
class Mesh_Ops(Ops):
    '''
    Object Responsible For:

    Initial Step of compiling functions for the following steps:
        - Apply BC and Cell Value
        - Face Interpolation 
        - Calculate Mass fluxes
        - Calculate Gradients

        the actual cell and face struct arrays are not stored in this object
    '''
    def __init__(self,cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        super().__init__(cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype,int_dtype)
    
    def init(self):
        # @wp.kernel
        # def _apply_BC_kernel(
        #                     face_values:wp.array2d(dtype=self.float_dtype),
        #                      face_gradients: wp.array2d(dtype=self.float_dtype),
        #                     face_struct:wp.array(dtype= self.face_struct),
        #                      boundary_condition:wp.array(dtype=self.face_properties.boundary_value.dtype),
        #                      gradient_condition:wp.array(dtype=self.face_properties.gradient_value.dtype),
        #                      boundary_ids: wp.array(dtype=self.face_properties.boundary_face_ids.dtype),
        #                      boundary_types:wp.array2d(dtype=self.face_properties.boundary_type.dtype)):
            
        #     i,j = wp.tid() # B,num_outputs
            
        #     face_id = boundary_ids[i]

        #     if boundary_types[i,j] == 1: # Dirichlet
        #         face_values[face_id,j] = boundary_condition[face_id][j]
        #         # wp.printf('%d %d %f %f \n',j,face_id,face_struct[face_id].values[j],boundary_condition[face_id][j])
        #     elif boundary_types[i,j] == 2: # Von neumann
        #         face_gradients[face_id,j] = gradient_condition[face_id][j]


        @wp.kernel
        def _set_initial_conditions_kernel(IC:wp.array2d(dtype=self.float_dtype),cell_values:wp.array2d(dtype=self.float_dtype),output_indices:wp.array(dtype=self.int_dtype)):
            i,output_idx = wp.tid() # C,O
            output = output_indices[output_idx]
            cell_values[i,output] = IC[i,output]
        
        # # @wp.kernel
        # # def _apply_cell_value_kernel(cell_values:wp.array2d(dtype=self.float_dtype),
        # #                              cell_struct:wp.array(dtype=self.cell_struct),
        # #                              fixed_value: wp.array(dtype=self.cell_properties.fixed_value.dtype)):
        # #     i,output = wp.tid() # C,O


        #     # if cell_struct[i].value_is_fixed[output]:
        #     #     cell_values[i,output] = fixed_value[i][output]
            
        # @wp.kernel
        # def _rhie_chow_correction_kernel(mass_fluxes:wp.array(dtype = self.float_dtype),
        #                                  cell_values:wp.array2d(dtype = self.float_dtype),
        #                                  face_values:wp.array2d(dtype = self.float_dtype),
        #                                  cell_gradients:wp.array2d(dtype=self.vector_type),
        #                                  D_face: wp.array(dtype=self.float_dtype),
        #                                  cell_structs:wp.array(dtype = self.cell_struct),
        #                                  face_structs:wp.array(dtype=self.face_struct),
        #                                  internal_face_ids:wp.array(dtype= self.face_properties.internal_face_ids.dtype),
        #                                  output_indices:wp.array(dtype=self.int_dtype)):
        #     i = wp.tid() # Loop through internal faces only
        #     p = 3
        #     # output = output_indices[output_idx]
        #     face_id = internal_face_ids[i]
            
        #     adjacent_cell_ids = face_structs[face_id].adjacent_cells
        #     owner_cell = cell_structs[adjacent_cell_ids[0]]
        #     neighbor_cell = cell_structs[adjacent_cell_ids[1]] 
        #     owner_face_idx = face_structs[face_id].cell_face_index[0]
            
        #     normal = owner_cell.face_normal[owner_face_idx]

        #     owner_grad = cell_gradients[owner_cell.id,p]
        #     neighbor_grad = cell_gradients[neighbor_cell.id,p]

        #     p_grad = (cell_values[neighbor_cell.id,p] - cell_values[owner_cell.id,p])
        #     p_grad_avg = owner_grad*face_structs[face_id].norm_distance[0] + neighbor_grad*face_structs[face_id].norm_distance[1]
        #     dist_to_face = owner_cell.cell_centroid_to_face_centroid[owner_face_idx]

        #     rhie_chow = (p_grad - wp.dot(p_grad_avg,dist_to_face))/wp.length(face_structs[face_id].cell_distance)
        #     vec_norm = wp.vector(face_values[face_id,0],face_values[face_id,1],face_values[face_id,2]) 
            
        #     v = wp.dot(vec_norm,normal)

        #     v = v -D_face[face_id]*rhie_chow
        #     mass_fluxes[face_id] = v*face_structs[face_id].area
            
            
            
        # @wp.kernel
        # def _boundary_calculate_face_interpolation_kernel(cell_values:wp.array2d(dtype=self.float_dtype),
        #                                                   face_values:wp.array2d(dtype=self.float_dtype),
        #                                                   face_gradients:wp.array2d(dtype=self.float_dtype),
        #                                                   face_structs:wp.array(dtype=self.face_struct),
        #                                   cell_structs:wp.array(dtype = self.cell_struct),
        #                                   boundary_face_ids:wp.array(dtype= self.int_dtype),
        #                                   boundary_type: wp.array(dtype= wp.uint8),
        #                                   output_indices:wp.array(dtype=self.int_dtype)):
            
        #     i,output_idx = wp.tid() # Loop through Boundary faces only
        #     output = output_indices[output_idx]
        #     face_id = boundary_face_ids[i]
        #     adjacent_cell_ids = face_structs[face_id].adjacent_cells
        #     owner_cell = cell_structs[adjacent_cell_ids[0]] # This is the only connected cell
        #     cell_face_idx = face_structs[face_id].cell_face_index[0]

        #     distance = wp.length(owner_cell.cell_centroid_to_face_centroid[cell_face_idx])
        #     if boundary_type[i][output] == 2: # Gradient is fixed
        #         face_values[face_id,output] = distance*face_gradients[face_id,output] + cell_values[owner_cell.id,output]    
            # if face_structs[face_id].gradient_is_fixed[output]: # We only do face interpolation if the gradient is fixed
            #     face_values[face_id,output] = distance*face_gradients[face_id,output] + cell_values[owner_cell.id,output]
                # wp.printf('%d %f %f %f %f \n',output,face_values[face_id,output],face_gradients[face_id,output],distance,cell_values[owner_cell.id,output])
        @wp.kernel
        def _calculate_mass_flux_kernel(mass_fluxes:wp.array(dtype=self.float_dtype),face_values:wp.array2d(dtype=self.float_dtype),cell_structs:wp.array(dtype = self.cell_struct),
                                  face_structs:wp.array(dtype = self.face_struct),
        ):
            face_id = wp.tid()
            face = face_structs[face_id]
            # i,j = wp.tid() # C,F
            owner_cell = cell_structs[face.adjacent_cells[0]] 
            normal = owner_cell.face_normal[face.cell_face_index[0]]
            area = face_structs[face_id].area
            #Compute dot product
            v = wp.vector(face_values[face_id,0],face_values[face_id,1],face_values[face_id,2])

            mass_fluxes[face_id] = wp.dot(v,normal)*area
            
        @wp.kernel
        def _calculate_gradients_kernel(face_values:wp.array2d(dtype=self.float_dtype),
                                        cell_gradients:wp.array2d(dtype = self.vector_type),
                                        cell_structs:wp.array(dtype = self.cell_struct),
                                        face_structs:wp.array(dtype = self.face_struct),
                                        node_structs:wp.array(dtype= self.node_struct),
                                        output_indices:wp.array(dtype=self.int_dtype),
                                        ):
            #Lets Use Gauss Linear from openFoam for now
            i,face_idx,output_idx = wp.tid() #C, faces_per_cell, num_outputs
            output = output_indices[output_idx]

            face_id = cell_structs[i].faces[face_idx]

            area = face_structs[face_id].area
            normal = cell_structs[i].face_normal[face_idx]
            volume = cell_structs[i].volume
            vec = face_values[face_id,output]*area*normal/volume
            # wp.printf('%f %f %f\n',vec[0],vec[1],vec[2])
            wp.atomic_add(cell_gradients,i,output,vec)

        @wp.kernel
        def _calculate_divergence_kernel(mass_fluxes:wp.array(dtype=self.float_dtype),cell_structs:wp.array(dtype=self.cell_struct),div_u:wp.array(dtype=self.float_dtype)):
            i,j = wp.tid() # C,F
            # calculate divergence
            face_id = cell_structs[i].faces[j]
            if cell_structs[i].face_sides[j] == 0: # Means cell is the Owner side of face
                wp.atomic_add(div_u,i,mass_fluxes[face_id])
            else: # means is neighbor side of face so set to -ve
                wp.atomic_add(div_u,i,-mass_fluxes[face_id])

        # self._apply_BC = _apply_BC_kernel
        # self._apply_cell_value = _apply_cell_value_kernel
        self._set_initial_conditions = _set_initial_conditions_kernel

        # self._internal_calculate_face_interpolation = _internal_calculate_face_interpolation_kernel
        # self._boundary_calculate_face_interpolation = _boundary_calculate_face_interpolation_kernel
        self._calculate_mass_flux = _calculate_mass_flux_kernel 
        self._calculate_gradients = _calculate_gradients_kernel
        # self._rhie_chow_correction = _rhie_chow_correction_kernel
        self._calculate_divergence = _calculate_divergence_kernel # mesh_ops

    def apply_BC(self,boundary_face_ids,boundary_types,face_values,face_gradients,faces): 
        wp.launch(kernel=self._apply_BC,dim = (boundary_face_ids.shape[0],self.num_outputs),inputs =[face_values,face_gradients,faces,self.face_properties.boundary_value,self.face_properties.gradient_value,boundary_face_ids,boundary_types])
    
    
    def set_initial_conditions(self,IC:wp.array,cell_values:wp.array,output_indices = None):
        if output_indices is None:
            output_indices = wp.array([i for i in range(IC.shape[-1])],dtype = self.int_dtype)
        wp.launch(kernel= self._set_initial_conditions, dim = [cell_values.shape[0],len(output_indices)],inputs= [
            IC,cell_values,output_indices])
        

    ''' Following are deprecated Functions'''
    # def apply_cell_value(self,cells):
    #     wp.launch(kernel=self._apply_cell_value,dim = (cells.shape[0],self.num_outputs) ,inputs = [cells,self.cell_properties.fixed_value])

    # def calculate_face_interpolation(self,mass_fluxes,cell_values,face_values,face_gradients,cells,faces,output_indices:wp.array | None = None,interpolation_method = 1):
    #     '''
    #     cell_values:wp.array2d(self.float_dtype),
    #                                                       face_values:wp.array2d(self.float_dtype),
    #                                                       face_gradients:wp.array2d(self.float_dtype)
    #     '''
        
    #     if output_indices is None:
    #         output_indices = wp.array([i for i in range(self.num_outputs)])

    #     wp.launch(kernel = self._internal_calculate_face_interpolation, dim = (self.face_properties.internal_face_ids.shape[0],output_indices.shape[0]), inputs = [mass_fluxes,cell_values,face_values,faces,cells,self.face_properties.internal_face_ids,output_indices,interpolation_method])
    #     wp.launch(kernel = self._boundary_calculate_face_interpolation, dim = (self.face_properties.boundary_face_ids.shape[0],output_indices.shape[0]), inputs = [cell_values,face_values,face_gradients,faces,cells,self.face_properties.boundary_face_ids,output_indices])
    
    # def rhie_chow_correction(self,mass_fluxes,cell_values,face_values,cell_gradients,D_face,cells,faces,output_indices = None):
    #     '''
    #     cell_values:wp.array2d(self.float_dtype),
    #                                      face_values:wp.array2d(self.float_dtype),
    #                                      cell_gradients:wp.array2d(dtype=self.vector_type)
    #                                       D_face: wp.array(dtype=self.float_dtype),
    #                                       cell_structs:wp.array(dtype = self.cell_struct),
    #                                       face_structs:wp.array(dtype=self.face_struct),
    #                                       internal_face_ids:wp.array(dtype= self.face_properties.internal_face_ids.dtype),
    #                                       output_indices:wp.array(dtype=self.int_dtype)
    #     '''
    #     if output_indices is None:
    #         output_indices = wp.array([0,1,2],dtype= self.int_dtype) 
    #     internal_face_ids= self.face_properties.internal_face_ids 
    #     wp.launch(self._rhie_chow_correction, dim = (internal_face_ids.shape[0]), inputs = [mass_fluxes,cell_values,face_values,cell_gradients,D_face,cells,faces,internal_face_ids,output_indices])


    def interpolate_internal_faces(self,mass_fluxes,cell_values,face_values,cells,faces,output_indices:wp.array | None = None,interpolation_method = 1):
        if output_indices is None:
            output_indices = wp.array([i for i in range(self.num_outputs)])
        wp.launch(kernel = self._internal_calculate_face_interpolation, dim = (self.face_properties.internal_face_ids.shape[0],output_indices.shape[0]), inputs = [mass_fluxes,cell_values,face_values,faces,cells,self.face_properties.internal_face_ids,output_indices,interpolation_method])
    def calculate_mass_flux(self,mass_fluxes,face_values,cells,faces):
        wp.launch(kernel = self._calculate_mass_flux,dim = (mass_fluxes.shape[0]),inputs = [mass_fluxes,face_values,cells,faces,])
    
    def calculate_gradients(self,face_values,cell_gradients:wp.array,cells,faces,nodes,output_indices):        
        wp.launch(kernel=self._calculate_gradients,dim = (cells.shape[0],self.faces_per_cell,len(output_indices)),inputs = [face_values,cell_gradients,cells,faces,nodes,output_indices])


    def calculate_divergence(self,mass_fluxes,cells :wp.array,arr:wp.array | None = None):
        if arr is None:
            arr = wp.zeros(shape = self.num_cells,dtype=self.float_dtype)
        arr.zero_()
        wp.launch(kernel=self._calculate_divergence, dim = (self.num_cells,self.faces_per_cell),inputs = [mass_fluxes,cells,arr])

        return arr
    
    def integrate_HbyA(self,field_arr,cells,faces,out = None | wp.array):
        if out is None:
            out = wp.zeros(shape = self.num_cells,dtype=self.float_dtype)
        else:
            out.zero_()



from warp.types import vector
@wp.kernel
def get_gradient_kernel(cell_array:wp.array2d(dtype = Any),coeff:wp.array(dtype=Any),cell_gradients:wp.array2d(dtype = Any),global_output_idx:int):
    i,j = wp.tid() # C,3
    if coeff.shape[0] == 1:
        coeff_ = coeff[0]
    else:
        coeff_ = coeff[i]
    cell_array[i,j] = coeff_*cell_gradients[i,global_output_idx][j]

# @wp.overload(get_gradient_kernel,[wp.array2d(dtype = wp.float32),wp.arraywp.float32,vector(3,dtype=wp.float32)])
# @wp.overload(get_gradient_kernel,[wp.array2d(dtype = wp.float64),wp.array(dtype = wp.float64),vector(3,dtype=wp.float64)])



def divFlux_kernel(cell_struct,float_dtype):
    @wp.kernel
    def _divFlux_kernel(mass_fluxes:wp.array(dtype=float_dtype),cell_structs:wp.array(dtype=cell_struct),div_u:wp.array(dtype=float_dtype)):
        i,j = wp.tid() # C,F
        # calculate divergence
        face_id = cell_structs[i].faces[j]
        if cell_structs[i].face_sides[j] == 0: # Means cell is the Owner side of face
            wp.atomic_add(div_u,i,mass_fluxes[face_id])
        else: # means is neighbor side of face so set to -ve
            wp.atomic_add(div_u,i,-mass_fluxes[face_id])
    return _divFlux_kernel
# @wp.kernel
# def integrate_divergence(arr:wp.array(dtype=float),cell_structs:wp.array(dtype=Any),face_structs:wp.array(dtype=Any),div_u:wp.array(dtype=float)):
#     i,j = wp.tid() # C,F
#             # calculate divergence

    
#     face_id = cell_structs[i].faces[j]

#     if cell_structs[i].face_sides[j] == 0: # Means cell is the Owner side of face
#         wp.atomic_add(div_u,i,arr*face_structs[face_id].area)
#     else: # means is neighbor side of face so set to -ve
#         wp.atomic_add(div_u,i,arr*face_structs[face_id].area)

