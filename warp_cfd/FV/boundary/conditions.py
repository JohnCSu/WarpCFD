import warp as wp
from typing import Any

from warp.types import vector
@wp.kernel
def apply_BC_kernel(
                    face_values:wp.array2d(dtype=Any),
                    face_gradients: wp.array2d(dtype=Any),
                    boundary_value:wp.array2d(dtype=Any),
                    boundary_ids: wp.array(dtype=wp.int32),
                    boundary_types:wp.array2d(dtype=wp.uint8)):

    i,j = wp.tid() # B,num_outputs
    boundary_id = boundary_ids[i]
    if boundary_types[i,j] == 1: # Dirichlet
        face_values[boundary_id,j] = boundary_value[i,j]
        # wp.printf('%d %d %f %f \n',j,face_id,face_struct[face_id].values[j],boundary_condition[face_id][j])
    elif boundary_types[i,j] == 2: # Von neumann
        face_gradients[boundary_id,j] = boundary_value[i,j]


for T in [wp.float32,wp.float64]:
    wp.overload(apply_BC_kernel,{'face_values':wp.array2d(dtype=T),
                                "face_gradients": wp.array2d(dtype=T),
                                "boundary_value":wp.array2d(dtype=T)})


@wp.kernel
def set_initial_conditions_kernel(IC:wp.array2d(dtype=Any),cell_values:wp.array2d(dtype=Any),output_indices:wp.array(dtype=wp.int32)):
    i,output_idx = wp.tid() # C,O
    output = output_indices[output_idx]
    cell_values[i,output] = IC[i,output]


for T in [wp.float32,wp.float64]:
    wp.overload(set_initial_conditions_kernel,{'IC':wp.array2d(dtype=T),'cell_values':wp.array2d(dtype=T)})

    