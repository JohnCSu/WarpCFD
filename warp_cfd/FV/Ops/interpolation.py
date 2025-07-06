import warp as wp
from typing import Any
@wp.func
def central_difference(cell_values:wp.array2d(dtype = Any),
                       cell_gradients:wp.array2d(dtype = Any),
                       owner_cell:Any,
                        neighbor_cell:Any,
                        face:Any,
                        output:int):
    psi = face.norm_distance[0]
    psi_2 = face.norm_distance[1] # Equal to (1-psi)
    return cell_values[owner_cell.id,output]*psi + psi_2*cell_values[neighbor_cell.id,output]


@wp.func
def upwind( cell_values:wp.array2d(dtype=Any),
           cell_gradients:wp.array2d(dtype = Any),
            owner_cell:Any,
            neighbor_cell:Any,
            face:Any,
            output:int):
    j = face.cell_face_index[0]
    if owner_cell.mass_fluxes[j] > 0:
        return cell_values[owner_cell.id,output]
    else:
        return cell_values[neighbor_cell.id,output]
    

@wp.func
def upwind_linear( cell_values:wp.array2d(dtype=Any),
           cell_gradients:wp.array2d(dtype = Any),
            owner_cell:Any,
            neighbor_cell:Any,
            face:Any,
            output:int):
    j = face.cell_face_index[0]
    if owner_cell.mass_fluxes[j] > 0:
        return cell_values[owner_cell.id,output] + wp.dot(owner_cell.cell_centroid_to_face_centroid[j],cell_gradients[owner_cell.id,output])
    else:
        k = face.cell_face_index[1]
        return cell_values[neighbor_cell.id,output] + wp.dot(neighbor_cell.cell_centroid_to_face_centroid[k],cell_gradients[neighbor_cell.id,output])
    

