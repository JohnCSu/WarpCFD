import warp as wp
from typing import Any
from .interpolation_functions import linear_interpolation,upwind


@wp.func
def skew_correction(cell_values:wp.array2d(dtype=Any),
                    cell_gradients:wp.array2d(dtype=Any),
           mass_fluxes:wp.array(dtype = Any),
            owner_cell:Any,
            neighbor_cell:Any,
            face:Any,
            global_output_idx:int):
    rk_dash = owner_cell.centroid*face.norm_distance[0] + neighbor_cell.centroid*face.norm_distance[1]
    rk_sub_rk_dash = face.centroid - rk_dash
    
    grad_interp = linear_interpolation(cell_gradients,mass_fluxes,owner_cell,neighbor_cell,face,global_output_idx)

    return wp.dot(grad_interp,rk_sub_rk_dash)
    # return 0.


