import warp as wp
from typing import Any
def central_difference(float_dtype):
    @wp.func
    def _central_difference(
                        cell_values:wp.array2d(dtype = float_dtype),
                        cell_gradients:wp.array2d(dtype = Any),
                        mass_fluxes:wp.array(dtype = float_dtype),
                        owner_cell:Any,
                            neighbor_cell:Any,
                            face:Any,
                            global_output_idx:int,
                            ):


        weight_vec = wp.vector(length=3,dtype=float_dtype)
        d = wp.length(face.cell_distance)
        #Owner Weight
        weight_vec[0] = float_dtype(-1.)/d
        #Neighbor Weight
        weight_vec[1] = float_dtype(1.)/d
        #Explicit Weight
        weight_vec[2] = float_dtype(0.)

        return weight_vec
    
    return _central_difference