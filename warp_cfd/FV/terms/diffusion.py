import warp as wp
from typing import Any
from warp_cfd.FV.finiteVolume import FVM
from warp_cfd.FV.terms.field import Field
from warp_cfd.FV.Implicit_Schemes.gradient_interpolation import central_difference
from warp_cfd.FV.terms.terms import Term

class DiffusionTerm(Term):
    def __init__(self,fv:FVM, field: Field| list[Field],interpolation='central difference',custom_interpolation = None):
        super().__init__(fv,field)
        self.interpolation = interpolation
        if interpolation == 'central difference':
            self.interpolation_function = central_difference
        else:
            assert custom_interpolation is not None, 'if interpolation is not central difference, a wp.func needs to be passed into custom_interpolation'

            self.interpolation_function  = custom_interpolation
        self._calculate_diffusion_weights_kernel = create_diffusion_scheme(self.interpolation_function,fv.cell_struct,fv.weight_struct,fv.face_struct,self.float_dtype,self.int_dtype)
    
    def calculate_weights(self,fv:FVM,viscosity:float | wp.array,**kwargs):
        if isinstance(viscosity,float):
            viscosity = wp.array([viscosity],dtype = fv.float_dtype)
        wp.launch(self._calculate_diffusion_weights_kernel,dim = [fv.faces.shape[0],self.num_outputs] ,inputs= [viscosity,fv.cell_values,fv.cell_gradients,fv.face_values,fv.face_gradients,fv.mass_fluxes,fv.cells,fv.faces,self.weights,self.global_output_indices])



def create_diffusion_scheme(interpolation_function,cell_struct,weight_struct,face_struct,float_dtype = wp.float32,int_dtype = wp.int32):
    @wp.kernel
    def diffusion_weights_kernel(viscosity:wp.array(dtype=float_dtype),
                                 cell_values:wp.array2d(dtype = float_dtype),
                                                cell_gradients:wp.array2d(dtype = Any),
                                                boundary_values:wp.array2d(dtype = float_dtype),
                                                boudary_gradients:wp.array2d(dtype = float_dtype),
                                                mass_fluxes:wp.array(dtype=float),
                                                cell_structs:wp.array(dtype = cell_struct),
                            face_structs:wp.array(dtype = face_struct),
                            weights:wp.array(ndim=3,dtype=weight_struct),
                            output_indices:wp.array(dtype=int_dtype),
                            ):
        
        face_id,output = wp.tid()

        global_var_idx = output_indices[output]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]
        neighbor_cell_id = face.adjacent_cells[1]

        if viscosity.shape[0] == 1:
            nu = viscosity[0]
        else:
            nu = viscosity[face_id]

        if face.is_boundary == 1:
            face_idx = face.cell_face_index[0]

            if face.gradient_is_fixed[output]:
                weights[owner_cell_id,face_idx,output].owner = 0. # Set the contribtuion to owner to 0 as for boundary term goes to RHS
                weights[owner_cell_id,face_idx,output].neighbor = 0.
                weights[owner_cell_id,face_idx,output].explicit_term = nu*boudary_gradients[face_id,global_var_idx]*face.area    
            else:
                cell_centroid_to_face_centroid_magnitude = wp.length(cell_structs[owner_cell_id].cell_centroid_to_face_centroid[face_idx])
                weight = nu*(face.area)/cell_centroid_to_face_centroid_magnitude
                
                weights[owner_cell_id,face_idx,output].owner = -weight # Set the contribtuion to owner to 0 as for boundary term goes to RHS
                weights[owner_cell_id,face_idx,output].neighbor = 0.
                weights[owner_cell_id,face_idx,output].explicit_term = weight*boundary_values[face_id,global_var_idx]

        else: # internal Faces
            distance = wp.length(face.cell_distance)
            weight = nu*(face.area)/distance
            face_indices = face.cell_face_index
            owner_cell = cell_structs[owner_cell_id]
            neighbor_cell = cell_structs[neighbor_cell_id]


            
            field_weighting = wp.static(interpolation_function)(cell_values,cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face,global_var_idx)

            #Owner, Neighbor , Explicit
            weights[owner_cell_id,face_indices[0],output].owner = field_weighting[0]*weight
            weights[owner_cell_id,face_indices[0],output].neighbor = field_weighting[1]*weight
            weights[owner_cell_id,face_indices[0],output].explicit_term = field_weighting[2]*weight

            # For neighbor we need to flip the mass flux first and then flip the weightings
            weights[neighbor_cell_id,face_indices[1],output].owner = field_weighting[1]*weight
            weights[neighbor_cell_id,face_indices[1],output].neighbor = field_weighting[0]*weight
            weights[neighbor_cell_id,face_indices[1],output].explicit_term = field_weighting[2]*weight
    return diffusion_weights_kernel

