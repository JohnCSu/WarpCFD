import warp as wp
from typing import Any
@wp.func
def central_difference(owner_cell:Any,
                        neighbor_cell:Any,
                        face:Any,
                        output:int):
    psi = face.norm_distance[0]
    psi_2 = face.norm_distance[1] # Equal to (1-psi)
    return owner_cell.values[output]*psi + psi_2*neighbor_cell.values[output]

@wp.func
def upwind(owner_cell:Any,
                        neighbor_cell:Any,
                        face:Any,
                        output:int):
    j = face.cell_face_index[0]
    if owner_cell.mass_fluxes[j] > 0:
        return owner_cell.values[output]
    else:
        return neighbor_cell.values[output]