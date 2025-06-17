from .interpolation import central_difference,upwind
import warp as wp
from typing import Any



class Mesh_Ops():
    def __init__(self,cell_struct,face_struct,node_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        self.cell_struct = cell_struct
        self.face_struct = face_struct
        self.node_struct = node_struct

        self.cell_properties = cell_properties 
        self.face_properties = face_properties
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.num_outputs = num_outputs
        self.faces_per_cell = cell_properties.faces_per_cell
        self.dimension= cell_properties.dimension


    def init_mesh_ops(self):
        @wp.kernel
        def _apply_BC_kernel(face_struct:wp.array(dtype= self.face_struct),
                             boundary_condition:wp.array(dtype=self.face_properties.boundary_value.dtype),
                             gradient_condition:wp.array(dtype=self.face_properties.gradient_value.dtype),
                             boundary_ids: wp.array(dtype=self.face_properties.boundary_face_ids.dtype)):
            
            i,j = wp.tid() # B,num_outputs
            
            face_id = boundary_ids[i]

            if face_struct[face_id].value_is_fixed[j] == 1:
                face_struct[face_id].values[j] = boundary_condition[face_id][j]
                # wp.printf('%d %d %f %f \n',j,face_id,face_struct[face_id].values[j],boundary_condition[face_id][j])
            if face_struct[face_id].gradient_is_fixed[j] == 1:
                face_struct[face_id].gradients[j] = gradient_condition[face_id][j]


        @wp.kernel
        def _apply_cell_value_kernel(cell_struct:wp.array(dtype=self.cell_struct),
                                     fixed_value: wp.array(dtype=self.cell_properties.fixed_value.dtype)):
            i,output = wp.tid() # C,O

            if cell_struct[i].value_is_fixed[output]:
                cell_struct[i].values[output] = fixed_value[i][output]
            
        @wp.kernel
        def _internal_calculate_face_interpolation_kernel(face_structs:wp.array(dtype=self.face_struct),
                                          cell_structs:wp.array(dtype = self.cell_struct),
                                          internal_face_ids:wp.array(dtype= self.face_properties.internal_face_ids.dtype),
                                          output_indices:wp.array(dtype=self.int_dtype),
                                          interpolation_method:int):
            
            '''
            We assume internal faces cannot have boundary conditions applied to them
            '''
            i,output_idx = wp.tid() # Loop through internal faces only
            output = output_indices[output_idx]
            face_id = internal_face_ids[i]
            adjacent_cell_ids = face_structs[face_id].adjacent_cells
            owner_cell = cell_structs[adjacent_cell_ids[0]]
            neighbor_cell = cell_structs[adjacent_cell_ids[1]] 

            if interpolation_method == 0: # Central Difference
                face_structs[face_id].values[output] = central_difference(owner_cell,neighbor_cell,face_structs[face_id],output)
            elif interpolation_method == 1: # Upwind
                face_structs[face_id].values[output] = upwind(owner_cell,neighbor_cell,face_structs[face_id],output)
            else:
                face_structs[face_id].values[output] = wp.nan
        @wp.kernel
        def _boundary_calculate_face_interpolation_kernel(face_structs:wp.array(dtype=self.face_struct),
                                          cell_structs:wp.array(dtype = self.cell_struct),
                                          boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
                                          output_indices:wp.array(dtype=self.int_dtype)):
            
            i,output_idx = wp.tid() # Loop through internal faces only
            output = output_indices[output_idx]
            face_id = boundary_face_ids[i]
            adjacent_cell_ids = face_structs[face_id].adjacent_cells
            owner_cell = cell_structs[adjacent_cell_ids[0]] # This is the only connected cell
            cell_face_idx = face_structs[face_id].cell_face_index[0]

            distance = wp.length(owner_cell.cell_centroid_to_face_centroid[cell_face_idx])

            if face_structs[face_id].gradient_is_fixed[output]: # We only do face interpolation if the gradient is fixed
                face_structs[face_id].values[output] = distance*face_structs[face_id].gradients[output] + owner_cell.values[output]

        @wp.kernel
        def _calculate_mass_flux_kernel(cell_structs:wp.array(dtype = self.cell_struct),
                                  face_structs:wp.array(dtype = self.face_struct),
        ):
            
            i,j = wp.tid() # C,F 
            face_id = cell_structs[i].faces[j]
            normal = cell_structs[i].face_normal[j]
            area = face_structs[face_id].area
            
            #Compute dot product
            dot_prod = wp.static(self.float_dtype(0.))
            for k in range(3):
              dot_prod += face_structs[face_id].values[k]*normal[k] 

            cell_structs[i].mass_fluxes[j] = dot_prod*area
            
        @wp.kernel
        def _calculate_gradients_kernel(cell_structs:wp.array(dtype = self.cell_struct),
                                        face_structs:wp.array(dtype = self.face_struct),
                                        node_structs:wp.array(dtype= self.node_struct),
                                        ):
            
            #Lets Use Gauss Linear from openFoam for now
            i = wp.tid() #C, faces_per_cell, num_outputs
            
            
            m = wp.matrix(shape= (wp.static(self.num_outputs),wp.static(self.dimension)),dtype= wp.float32)

            for face_idx in range(wp.static(self.faces_per_cell)):
                face_id = cell_structs[i].faces[face_idx]
                normal = cell_structs[i].face_normal[face_idx]
                m += wp.outer(face_structs[face_id].values,normal)*face_structs[face_id].area/cell_structs[i].volume
            cell_structs[i].gradients = m


        self._apply_BC = _apply_BC_kernel
        self._apply_cell_value = _apply_cell_value_kernel
        self._internal_calculate_face_interpolation = _internal_calculate_face_interpolation_kernel
        self._boundary_calculate_face_interpolation = _boundary_calculate_face_interpolation_kernel
        self._calculate_mass_flux = _calculate_mass_flux_kernel 
        self._calculate_gradients = _calculate_gradients_kernel

    def apply_BC(self):
        faces = self.face_properties
        wp.launch(kernel=self._apply_BC,dim = (self.face_properties.boundary_face_ids.shape[0],self.num_outputs),inputs =[self.faces,faces.boundary_value,faces.gradient_value,faces.boundary_face_ids])
    
    def apply_cell_value(self):
        wp.launch(kernel=self._apply_cell_value,dim = (self.num_cells,self.num_outputs) ,inputs = [self.cells,self.cell_properties.fixed_value])

    def calculate_face_interpolation(self,output_indices:wp.array | None = None,interpolation_method = 1):
        if output_indices is None:
            output_indices = wp.array([i for i in range(self.num_outputs)])
        wp.launch(kernel = self._internal_calculate_face_interpolation, dim = (self.face_properties.internal_face_ids.shape[0],output_indices.shape[0]), inputs = [self.faces,self.cells,self.face_properties.internal_face_ids,output_indices,interpolation_method])
        wp.launch(kernel = self._boundary_calculate_face_interpolation, dim = (self.face_properties.boundary_face_ids.shape[0],output_indices.shape[0]), inputs = [self.faces,self.cells,self.face_properties.boundary_face_ids,output_indices])
    
    
    def calculate_mass_flux(self):
        wp.launch(kernel = self._calculate_mass_flux,dim = (self.num_cells,self.faces_per_cell),inputs = [self.cells,
                                                                              self.faces,
                                                                              ])
    
    def calculate_gradients(self):
        wp.launch(kernel=self._calculate_gradients,dim = (self.num_cells),inputs = [self.cells,self.faces,self.nodes])
