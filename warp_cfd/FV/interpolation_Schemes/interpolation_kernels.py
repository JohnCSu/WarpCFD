import warp as wp
from typing import Any
from .skewness_functions import skew_correction

def internal_calculate_face_interpolation_kernel(interpolation_function,cell_struct,face_struct,skew_corrected,float_dtype):
    @wp.kernel
    def _internal_calculate_face_interpolation_kernel(cell_values:wp.array2d(dtype = float_dtype),
                                                        cell_gradients:wp.array2d(dtype = Any),
                                                        mass_fluxes:wp.array(dtype = float_dtype),
                                                        face_values:wp.array2d(dtype=float_dtype),
                                                        face_structs:wp.array(dtype=face_struct),
                                        cell_structs:wp.array(dtype = cell_struct),
                                        internal_face_ids:wp.array(dtype= wp.int32),
                                        output_indices:wp.array(dtype=wp.int32)):
        
        '''
        We assume internal faces cannot have boundary conditions applied to them
        '''
        i,output_idx = wp.tid() # Loop through internal faces only
        output = output_indices[output_idx]
        face_id = internal_face_ids[i]
        adjacent_cell_ids = face_structs[face_id].adjacent_cells
        owner_cell = cell_structs[adjacent_cell_ids[0]]
        neighbor_cell = cell_structs[adjacent_cell_ids[1]] 

        face_values[face_id,output] = wp.static(interpolation_function)(cell_values,mass_fluxes,owner_cell,neighbor_cell,face_structs[face_id],output)
        if wp.static(skew_corrected):
            face_values[face_id,output] += skew_correction(cell_values,cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face_structs[face_id],output)


    return _internal_calculate_face_interpolation_kernel

def boundary_calculate_face_interpolation_kernel(cell_struct,face_struct,float_dtype):
    @wp.kernel
    def _boundary_calculate_face_interpolation_kernel(cell_values:wp.array2d(dtype=float_dtype),
                                                        face_values:wp.array2d(dtype=float_dtype),
                                                        face_gradients:wp.array2d(dtype=float_dtype),
                                                        face_structs:wp.array(dtype=face_struct),
                                        cell_structs:wp.array(dtype = cell_struct),
                                        boundary_face_ids:wp.array(dtype= wp.int32),
                                        boundary_types: wp.array2d(dtype= wp.uint8),
                                        output_indices:wp.array(dtype=wp.int32)):
        
        i,output_idx = wp.tid() # Loop through Boundary faces only
        output = output_indices[output_idx]
        face_id = boundary_face_ids[i]
        adjacent_cell_ids = face_structs[face_id].adjacent_cells
        owner_cell = cell_structs[adjacent_cell_ids[0]] # This is the only connected cell
        cell_face_idx = face_structs[face_id].cell_face_index[0]

        distance = wp.length(owner_cell.cell_centroid_to_face_centroid[cell_face_idx])

        if boundary_types[i,output] == 2 : # We only do face interpolation if the gradient is fixed
            face_values[face_id,output] = distance*face_gradients[face_id,output] + cell_values[owner_cell.id,output]
    return _boundary_calculate_face_interpolation_kernel