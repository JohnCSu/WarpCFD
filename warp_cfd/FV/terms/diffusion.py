import warp as wp
from typing import Any
from warp_cfd.FV.model import FVM
from warp_cfd.FV.field import Field
from warp_cfd.FV.Implicit_Schemes.gradient_interpolation import central_difference
from warp_cfd.FV.terms.terms import Term

class DiffusionTerm(Term):
    def __init__(self,fv:FVM, fields: Field| list[Field],*,interpolation='central difference',custom_interpolation = None,von_neumann = None, dirchlet =None,need_global_index= True):
        super().__init__(fv,fields,implicit= True,need_global_index = need_global_index)
        #Define interpolation function
        self.interpolation = interpolation
        if interpolation == 'central difference':
            self.interpolation_function = central_difference(fv.float_dtype)
        else:
            assert custom_interpolation is not None, 'if interpolation is not central difference, a wp.func needs to be passed into custom_interpolation'

            self.interpolation_function  = custom_interpolation

        # Implicit Diffusion Kernel
        self._calculate_internal_diffusion_weights_kernel = create_diffusion_scheme(self.interpolation_function,fv.cell_struct,fv.face_struct,self.float_dtype,self.int_dtype)

        # Define the BC kernel
        assert not ((von_neumann is not None) and (dirchlet is not None)), 'von neumann and dirichlet settings cannot both be not None'
        
        
        
        if von_neumann is not None:
            assert len(self.fields) == 1 ,'Von neumann or dirichlet option only valid if len of field == 1'
            
            self._calculate_boundary_diffusion_weights_kernel = create_von_neumann_BC_diffusion_kernel(von_neumann,fv.cell_struct,fv.face_struct,self.float_dtype,self.int_dtype)
        elif dirchlet is not None:
            assert len(self.fields) == 1 ,'Von neumann or dirichlet option only valid if len of field == 1'
            self._calculate_boundary_diffusion_weights_kernel = create_von_neumann_BC_diffusion_kernel(dirchlet,fv.cell_struct,fv.face_struct,self.float_dtype,self.int_dtype)
        else:
            self._calculate_boundary_diffusion_weights_kernel = creating_diffusion_BC_scheme_kernel(fv.cell_struct,fv.face_struct,self.float_dtype,self.int_dtype)

    
    def calculate_weights(self,fv:FVM,viscosity:float | wp.array,**kwargs):
        if isinstance(viscosity,float):
            viscosity = wp.array([viscosity],dtype = fv.float_dtype)

        boundary_face_ids = fv.face_properties.boundary_face_ids
        internal_face_ids = fv.face_properties.internal_face_ids
        wp.launch(self._calculate_boundary_diffusion_weights_kernel,dim = [boundary_face_ids.shape[0],self.num_outputs], inputs = [boundary_face_ids,viscosity,fv.face_values,fv.face_gradients,fv.cells,fv.faces,self.weights,self.global_output_indices])
        wp.launch(self._calculate_internal_diffusion_weights_kernel,dim = [internal_face_ids.shape[0],self.num_outputs] ,inputs= [internal_face_ids,viscosity,fv.cell_values,fv.cell_gradients,fv.mass_fluxes,fv.cells,fv.faces,self.weights,self.global_output_indices])




def create_von_neumann_BC_diffusion_kernel(von_neumann_value,cell_struct,face_struct,float_dtype = wp.float32,int_dtype = wp.int32):
    
    von_neumann_value = float_dtype(von_neumann_value)
    @wp.kernel
    def von_neumann_BC_diffusion_kernel(boundary_ids:wp.array(dtype=int),
                                        viscosity:wp.array(dtype=float_dtype),
                                        boundary_values:wp.array2d(dtype = float_dtype),
                                        boudary_gradients:wp.array2d(dtype = float_dtype),
                                        cell_structs:wp.array(dtype = cell_struct),
                                        face_structs:wp.array(dtype = face_struct),
                                        weights:wp.array(ndim=4,dtype=float_dtype),
                                        output_indices:wp.array(dtype=int_dtype)):
        i,_ = wp.tid()
        face_id = boundary_ids[i]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]
        face_idx = face.cell_face_index[0]

        if viscosity.shape[0] == 1:
            nu = viscosity[0]
        else:
            nu = viscosity[face_id]
        
        weights[owner_cell_id,face_idx,0,0] = float_dtype(0.) # Set the contribtuion to owner to 0 as for boundary term goes to RHS
        weights[owner_cell_id,face_idx,0,1] = float_dtype(0.)
        weights[owner_cell_id,face_idx,0,2] = nu*von_neumann_value*face.area    

    return von_neumann_BC_diffusion_kernel

def create_dirichlet_BC_kernel(dirichlet_value,cell_struct,face_struct,float_dtype = wp.float32,int_dtype = wp.int32):
    @wp.kernel
    def dirichlet_BC_kernel_BC_diffusion_kernel(boundary_ids:wp.array(dtype=int),
                                        viscosity:wp.array(dtype=float_dtype),
                                        boundary_values:wp.array2d(dtype = float_dtype),
                                        boudary_gradients:wp.array2d(dtype = float_dtype),        
                                        cell_structs:wp.array(dtype = cell_struct),
                                        face_structs:wp.array(dtype = face_struct),
                                        weights:wp.array(ndim=4,dtype=float_dtype),
                                        output_indices:wp.array(dtype=int_dtype)):
        i,_ = wp.tid()
        face_id = boundary_ids[i]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]
        face_idx = face.cell_face_index[0]
        if viscosity.shape[0] == 1:
            nu = viscosity[0]
        else:
            nu = viscosity[face_id]
        
        cell_centroid_to_face_centroid_magnitude = wp.length(cell_structs[owner_cell_id].cell_centroid_to_face_centroid[face_idx])
        weight = nu*(face.area)/cell_centroid_to_face_centroid_magnitude
        
        weights[owner_cell_id,face_idx,0,0] = -weight # Set the contribtuion to owner to 0 as for boundary term goes to RHS
        weights[owner_cell_id,face_idx,0,1] = float_dtype(0.)
        weights[owner_cell_id,face_idx,0,2] = weight*dirichlet_value
    
    return dirichlet_BC_kernel_BC_diffusion_kernel


def creating_diffusion_BC_scheme_kernel(cell_struct,face_struct,float_dtype = wp.float32,int_dtype = wp.int32):
    
    @wp.kernel
    def diffusion_BC_kernel(boundary_ids:wp.array(dtype=int),
                            viscosity:wp.array(dtype=float_dtype),
                            boundary_values:wp.array2d(dtype = float_dtype),
                            boudary_gradients:wp.array2d(dtype = float_dtype),
                            cell_structs:wp.array(dtype = cell_struct),
                            face_structs:wp.array(dtype = face_struct),
                            weights:wp.array(ndim=4,dtype=float_dtype),
                            output_indices:wp.array(dtype=int_dtype),):
        
        i,output = wp.tid()
        face_id = boundary_ids[i]
        global_var_idx = output_indices[output]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]

        if viscosity.shape[0] == 1:
            nu = viscosity[0]
        else:
            nu = viscosity[face_id]

    
        face_idx = face.cell_face_index[0]

        if face.gradient_is_fixed[global_var_idx]:
            weights[owner_cell_id,face_idx,output,0] = float_dtype(0.) # Set the contribtuion to owner to 0 as for boundary term goes to RHS
            weights[owner_cell_id,face_idx,output,1] = float_dtype(0.)
            weights[owner_cell_id,face_idx,output,2] = nu*boudary_gradients[face_id,global_var_idx]*face.area    
        else:
            cell_centroid_to_face_centroid_magnitude = wp.length(cell_structs[owner_cell_id].cell_centroid_to_face_centroid[face_idx])
            weight = nu*(face.area)/cell_centroid_to_face_centroid_magnitude
            
            weights[owner_cell_id,face_idx,output,0] = -weight # Set the contribtuion to owner to 0 as for boundary term goes to RHS
            weights[owner_cell_id,face_idx,output,1] = float_dtype(0.)
            weights[owner_cell_id,face_idx,output,2] = weight*boundary_values[face_id,global_var_idx]

    return diffusion_BC_kernel


def create_diffusion_scheme(interpolation_function,cell_struct,face_struct,float_dtype = wp.float32,int_dtype = wp.int32):
    @wp.kernel
    def diffusion_weights_kernel(internal_face_ids:wp.array(dtype= int),
                                 viscosity:wp.array(dtype=float_dtype),
                                 cell_values:wp.array2d(dtype = float_dtype),
                                cell_gradients:wp.array2d(dtype = Any),
                                mass_fluxes:wp.array(dtype=float_dtype),
                                cell_structs:wp.array(dtype = cell_struct),
                                face_structs:wp.array(dtype = face_struct),
                                weights:wp.array(ndim=4,dtype=float_dtype),
                                output_indices:wp.array(dtype=int_dtype),
                            ):
        
        i,output = wp.tid()

        face_id = internal_face_ids[i]

        global_var_idx = output_indices[output]
        face = face_structs[face_id]
        owner_cell_id = face.adjacent_cells[0]
        neighbor_cell_id = face.adjacent_cells[1]

        if viscosity.shape[0] == 1:
            nu = viscosity[0]
        else:
            nu = viscosity[face_id]
    
        weight = nu*(face.area)
        face_indices = face.cell_face_index
        owner_cell = cell_structs[owner_cell_id]
        neighbor_cell = cell_structs[neighbor_cell_id]

        field_weighting = wp.static(interpolation_function)(cell_values,cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face,global_var_idx)

        #Owner, Neighbor , Explicit
        weights[owner_cell_id,face_indices[0],output,0] = field_weighting[0]*weight
        weights[owner_cell_id,face_indices[0],output,1] = field_weighting[1]*weight
        weights[owner_cell_id,face_indices[0],output,2] = field_weighting[2]*weight

        # For neighbor we need to flip the mass flux first and then flip the weightings
        weights[neighbor_cell_id,face_indices[1],output,0] = -field_weighting[1]*weight
        weights[neighbor_cell_id,face_indices[1],output,1] = -field_weighting[0]*weight
        weights[neighbor_cell_id,face_indices[1],output,2] = -field_weighting[2]*weight
    return diffusion_weights_kernel

