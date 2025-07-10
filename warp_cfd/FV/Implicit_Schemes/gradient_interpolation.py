import warp as wp
from typing import Any

@wp.func
def central_difference(
                       cell_values:wp.array2d(dtype = float),
                       cell_gradients:wp.array2d(dtype = Any),
                       mass_fluxes:wp.array(dtype = float),
                       owner_cell:Any,
                        neighbor_cell:Any,
                        face:Any,
                        global_output_idx:int,
                        ):


    weight_vec = wp.vector(length=3,dtype=float)
    d = wp.length(face.cell_distance)
    #Owner Weight
    weight_vec[0] = -1./d
    #Neighbor Weight
    weight_vec[1] = 1./d
    #Explicit Weight
    weight_vec[2] = 0.

    return weight_vec