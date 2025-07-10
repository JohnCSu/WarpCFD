import warp as wp
from typing import Any

@wp.func
def central_difference(
                       cell_values:wp.array2d(dtype = float),
                       cell_gradients:wp.array2d(dtype = float),
                       mass_fluxes:wp.array(dtype = float),
                       owner_cell:Any,
                        neighbor_cell:Any,
                        face:Any,
                        output:int,
                        ):

    psi = face.norm_distance[0]
    psi_2 = face.norm_distance[1] # Equal to (1-psi)
    weight_vec = wp.vector(length=3,dtype=float)
    #Owner Weight
    weight_vec[0] = psi
    #Neighbor Weight
    weight_vec[1] = psi_2
    #Explicit Weight
    weight_vec[2] = 0.

    return weight_vec

@wp.func
def upwind( 
            cell_values:wp.array2d(dtype = float),
            cell_gradients:wp.array2d(dtype = Any),
            mass_fluxes:wp.array(dtype = float),
            owner_cell:Any,
            neighbor_cell:Any,
            face:Any,
            output:int,
            ):
    # 0 -> Owner, 1 -> Neighbor, 2 -> Explicit RHS term
    weight_vec = wp.vector(length=3,dtype=float)

    if mass_fluxes[face.id] > 0:
        weight_vec[0]= 1.
        weight_vec[1] = 0.
    else:
        weight_vec[0]= 0.
        weight_vec[1]= 1.
    
    weight_vec[2] = 0.

    return weight_vec
