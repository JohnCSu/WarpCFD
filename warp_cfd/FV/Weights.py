import warp as wp
from typing import Any

def create_weight_struct(float_dtype = wp.float32):
    @wp.struct
    class Weights:
        owner: float_dtype 
        neighbor: float_dtype
        # explicit: float_dtype
    return Weights



# @wp.func
# def central_difference(weight:Any,
#                        cell_values:wp.array2d(dtype = float),
#                        cell_gradients:wp.array2d(dtype = float),
#                        mass_fluxes:wp.array(dtype = float),
#                        owner_cell:Any,
#                         neighbor_cell:Any,
#                         face:Any,
#                         output:int,
#                         ):

#     psi = face.norm_distance[0]
#     psi_2 = face.norm_distance[1] # Equal to (1-psi)

#     #Owner Weight
#     weight.owner = psi
#     #Neighbor Weight
#     weight.neighbor = psi_2
#     #Explicit Weight
#     weight.explicit = 0.

#     return weight

# @wp.func
# def upwind( weight:Any,
#             cell_values:wp.array2d(dtype = float),
#             cell_gradients:wp.array2d(dtype = float),
#             mass_fluxes:wp.array(dtype = float),
#             owner_cell:Any,
#             neighbor_cell:Any,
#             face:Any,
#             output:int,
#             ):

#     if mass_fluxes[face.id] > 0:
#         weight.owner= 1.
#         weight.neighbor = 0.
#     else:
#         weight.owner= 0.
#         weight.neighbor= 1.
    
#     weight.explicit = 0.

#     return weight



    

