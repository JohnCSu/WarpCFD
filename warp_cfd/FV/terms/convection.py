import warp as wp
from typing import Any
from warp_cfd.FV.model import FVM
from warp_cfd.FV.field import Field
from warp_cfd.FV.terms.terms import Term
from warp_cfd.FV.implicit_Schemes.convectionScheme import central_difference,upwind,upwindLinear
from warp_cfd.FV.interpolation_Schemes import skew_correction
class ConvectionTerm(Term):
    def __init__(self,fv:FVM, fields: Field| list[Field],interpolation='upwind',custom_interpolation = None):
        super().__init__(fv,fields, implicit = True, need_global_index = True,cell_based = False)

        self.interpolation_functions = {
            'upwind': upwind(fv.float_dtype),
            'central difference':central_difference(fv.float_dtype),
            'upwindLinear':upwindLinear(fv.float_dtype), 
            'custom':None # To keep assertion check easy
        }
        self.correct_face_interpolation = True
        self.interpolation = interpolation
        
        assert interpolation in self.interpolation_functions.keys(), 'interpolation schems supported: upwind, central difference, upwindLinear or custom'
        if interpolation == 'custom':
            assert custom_interpolation is not None # Should be better check here
            self.interpolation_functions['custom'] = custom_interpolation
            
        self.interpolation_function = self.interpolation_functions[interpolation]
        self._calculate_convection_weights_kernel = create_convection_scheme(self.interpolation_function,fv.cell_struct,fv.face_struct,self.correct_face_interpolation,self.float_dtype,self.int_dtype)



    def calculate_weights(self,fv:FVM,**kwargs):
        wp.launch(self._calculate_convection_weights_kernel,dim = [fv.faces.shape[0],self.num_outputs] ,inputs= [fv.cell_values,
                                                                                                                 fv.cell_gradients,
                                                                                                                 fv.face_values,
                                                                                                                 fv.mass_fluxes,
                                                                                                                 fv.cells,
                                                                                                                 fv.faces,
                                                                                                                 self.weights,
                                                                                                                 self.global_output_indices])



def create_convection_scheme(interpolation_function,cell_struct,face_struct,correct_face_interp,float_dtype = wp.float32,int_dtype = wp.int32):
    @wp.kernel
    def convection_weights_kernel(cell_values:wp.array2d(dtype = float_dtype),
                                                cell_gradients:wp.array2d(dtype = Any),
                                                face_values:wp.array2d(dtype = float_dtype),
                                                mass_fluxes:wp.array(dtype=float_dtype),
                                                cell_structs:wp.array(dtype = cell_struct),
                            face_structs:wp.array(dtype = face_struct),
                            weights:wp.array(ndim=4,dtype=float_dtype),
                            output_indices:wp.array(dtype=int_dtype)):
        '''
        To DO:
        Change Weights to be of size number of faces so we dont double up??
        '''
        face_id,output = wp.tid()

        global_var_idx = output_indices[output]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]
        neighbor_cell_id = face.adjacent_cells[1]
        if face.is_boundary == 1:
            face_idx = face.cell_face_index[0]
            weights[owner_cell_id,face_idx,output,0] = float_dtype(0.) # Set the contribtuion to owner to 0 as for boundary term goes to RHS
            weights[owner_cell_id,face_idx,output,1] = float_dtype(0.)
            weights[owner_cell_id,face_idx,output,2]= mass_fluxes[face_id]*face_values[face_id,global_var_idx]

        else: # internal Faces

            face_indices = face.cell_face_index
            owner_cell = cell_structs[owner_cell_id]
            neighbor_cell = cell_structs[neighbor_cell_id]
            
            field_weighting = wp.static(interpolation_function)(cell_values,cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face,global_var_idx)
            if wp.static(correct_face_interp):
                skew = skew_correction(cell_values,cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face_structs[face_id],output)
                field_weighting[2]+= skew
            # #Owner, Neighbor , Explicit
            weights[owner_cell_id,face_indices[0],output,0] = field_weighting[0]*mass_fluxes[face_id]
            weights[owner_cell_id,face_indices[0],output,1] = field_weighting[1]*mass_fluxes[face_id]
            weights[owner_cell_id,face_indices[0],output,2] = field_weighting[2]*mass_fluxes[face_id]

            # # For neighbor we need to flip the mass flux first and then flip the weightings
            mass_flux = -mass_fluxes[face_id]
            weights[neighbor_cell_id,face_indices[1],output,0] = field_weighting[1]*mass_flux
            weights[neighbor_cell_id,face_indices[1],output,1] = field_weighting[0]*mass_flux
            weights[neighbor_cell_id,face_indices[1],output,2]= field_weighting[2]*mass_flux

                   
    return convection_weights_kernel