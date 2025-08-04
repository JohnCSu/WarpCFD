import warp as wp
import numpy as np
from .ops_class import Ops


class Weights_Ops(Ops):
    def __init__(self,cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        super().__init__(cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype,int_dtype)


    def init(self):
        '''
        - Calculate Convection And Laplacian Weights
        '''
        @wp.kernel
        def _calculate_convection_weights_kernel(mass_fluxes:wp.array(dtype = self.float_dtype),face_values:wp.array2d(dtype = self.float_dtype),
                                                 cell_structs:wp.array(dtype = self.cell_struct),
                                face_structs:wp.array(dtype = self.face_struct),
                                weights:wp.array(ndim=3,dtype=self.weight_struct),
                                outputs:wp.array(dtype=self.int_dtype),
                                interpolation:int
                                ):
            
            i,j,k = wp.tid() # loop through C,F,O
            output = outputs[k]
            face_id = cell_structs[i].faces[j]
            
            if face_structs[face_id].is_boundary == 1:
                weights[i,j,output].owner = 0. # Set the contribtuion to owner to 0 as for boundary term goes to RHS
                weights[i,j,output].neighbor = mass_fluxes[face_id]*face_values[face_id,output]
            else:
                owner_face_id = cell_structs[i].face_sides[j] # Returns if current cell i is on the 0 side of face or 1 side of face
                adj_face_id = wp.static(self.int_dtype(1)) - owner_face_id # Apply not operation to get the other index (can only be 1 or 0)
                
                if interpolation == 0: # Central Differencing
                    weights[i,j,output].owner = face_structs[face_id].norm_distance[owner_face_id]*mass_fluxes[face_id]
                    weights[i,j,output].neighbor = face_structs[face_id].norm_distance[adj_face_id]*mass_fluxes[face_id]
                elif interpolation == 1: # Upwind

                    if cell_structs[i].face_sides[j] == 0: #Owner
                        mass_flux = mass_fluxes[face_id]

                    else:
                        mass_flux = -mass_fluxes[face_id]

                
                    if mass_flux > 0:
                        weights[i,j,output].owner = mass_flux
                        weights[i,j,output].neighbor = 0.
                    else:
                        weights[i,j,output].owner = 0.
                        weights[i,j,output].neighbor = mass_flux
                    
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
        def _calculate_laplacian_weights_kernel(face_values:wp.array2d(dtype = self.float_dtype),
                                                face_gradients:wp.array2d(dtype = self.float_dtype),
                                                cell_structs:wp.array(dtype = self.cell_struct),
                                face_structs:wp.array(dtype = self.face_struct),
                                weights:wp.array3d(dtype=self.weight_struct),
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
                    weights[i,j,output].owner = 0.
                    weights[i,j,output].neighbor = nu*face_structs[face_id].area*face_gradients[face_id,output]
                else:
                    cell_centroid_to_face_centroid_magnitude = wp.length(cell_structs[i].cell_centroid_to_face_centroid[j])
                    weight = nu*(face_structs[face_id].area)/cell_centroid_to_face_centroid_magnitude

                    weights[i,j,output].owner = -weight # Unknown 
                    weights[i,j,output].neighbor = weight*( face_values[face_id,output]) # Known Value so evaluate explicit
                # wp.printf('scalar %f nu %f area %f dist %f val %f  \n ',vis_area_div_dist,nu,face_structs[face_id].area,cell_centroid_to_face_centroid_magnitude,face_structs[face_id].values[output])

                # wp.printf('a %f fv %f cv %f \n',face_structs[face_id].area,face_structs[face_id].values[output],cell_structs[i].values[output])
            else:
                distance = wp.length(face_structs[face_id].cell_distance)
                weight = nu*(face_structs[face_id].area)/distance
                weights[i,j,output].owner = -weight
                weights[i,j,output].neighbor = weight
                # wp.printf('%f neighbor \n',weights[i,j,output].laplacian_neighbor)

        self._calculate_convection_weights = _calculate_convection_weights_kernel
        self._calculate_laplacian_weights = _calculate_laplacian_weights_kernel
        self._interpolate_viscosity_to_face = _interpolate_viscosity_to_face_kernel

    
    def calculate_convection(self,mass_fluxes,face_values,cells,faces,weights,output_indices:wp.array,interpolation_method = 1):
        num_outputs = len(output_indices)
        num_faces = faces.shape[0]
        num_cells = cells.shape[0]
        
        wp.launch(kernel=self._calculate_convection_weights,dim = (num_cells,self.faces_per_cell,num_outputs),inputs = [mass_fluxes,face_values,cells,faces,weights,output_indices,interpolation_method])
    

    def calculate_laplacian(self,face_values,face_gradients,cells,faces,weights,output_indices:wp.array,viscosity: float | wp.array):
        num_outputs = len(output_indices)
        num_faces = faces.shape[0]
        num_cells = cells.shape[0]
        if isinstance(viscosity,float):
            viscosity = wp.array([viscosity],dtype =self.float_dtype)

        elif isinstance(viscosity,wp.array):
            assert len(viscosity.shape) == 1
            if viscosity.shape[0] == num_cells:
                if self.face_viscosity is None:
                    self.face_viscosity = wp.zeros(shape = faces.shape[0],dtype= self.float_dtype)

                wp.launch(self._interpolate_viscosity_to_face,dim = faces.shape[0], inputs = [viscosity,self.face_viscosity,faces])
            
                viscosity = self.face_viscosity
            else:
                assert viscosity.shape[0] == num_faces
        else:
            raise ValueError('viscosity can be type float or wp.array')

        wp.launch(kernel=self._calculate_laplacian_weights,dim = (num_cells,self.faces_per_cell,num_outputs),inputs = [face_values,face_gradients,cells,faces,weights,viscosity,output_indices])
