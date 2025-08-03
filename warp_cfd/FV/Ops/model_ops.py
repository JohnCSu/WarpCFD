import warp as wp
from typing import Any
from warp_cfd.FV.mesh_structs import CELL_DICT


from warp.types import vector
@wp.kernel
def get_gradient_kernel(cell_array:wp.array2d(dtype = Any),coeff:wp.array(dtype=Any),cell_gradients:wp.array2d(dtype = Any),global_output_idx:int):
    i,j = wp.tid() # C,3
    if coeff.shape[0] == 1:
        coeff_ = coeff[0]
    else:
        coeff_ = coeff[i]
    cell_array[i,j] = coeff_*cell_gradients[i,global_output_idx][j]


for float_type in [wp.float32,wp.float64]:
    wp.overload(get_gradient_kernel,{"cell_array":wp.array2d(dtype = float_type),
                                     "coeff":wp.array(dtype= float_type),
                                     "cell_gradients":wp.array2d(dtype=vector(3,dtype=float_type)) })





@wp.kernel
def divFlux_kernel(mass_fluxes:wp.array(dtype=Any),cell_structs:wp.array(dtype=Any),div_u:wp.array(dtype=Any)):
    i,j = wp.tid() # C,F
    # calculate divergence
    face_id = cell_structs[i].faces[j]
    if cell_structs[i].face_sides[j] == 0: # Means cell is the Owner side of face
        wp.atomic_add(div_u,i,mass_fluxes[face_id])
    else: # means is neighbor side of face so set to -ve
        wp.atomic_add(div_u,i,-mass_fluxes[face_id])


for key,(cell_struct,face_struct,node_struct) in CELL_DICT.items():
    float_type = wp.float32 if 'F32' in key else wp.float64
    wp.overload(divFlux_kernel,{"mass_fluxes":wp.array(dtype=float_type),
                                        "cell_structs":wp.array(dtype=cell_struct),
                                        "div_u":wp.array(dtype=float_type) })



wp.vec3d
@wp.kernel
def calculate_mass_flux_kernel(mass_fluxes:wp.array(dtype=Any),
                               face_values:wp.array2d(dtype=Any),
                               cell_structs:wp.array(dtype = Any),
                            face_structs:wp.array(dtype = Any),
):
    face_id = wp.tid()
    face = face_structs[face_id]
    # i,j = wp.tid() # C,F
    owner_cell = cell_structs[face.adjacent_cells[0]] 
    normal = owner_cell.face_normal[face.cell_face_index[0]]
    area = face_structs[face_id].area
    #Compute dot product
    v = wp.vector(face_values[face_id,0],face_values[face_id,1],face_values[face_id,2])

    mass_fluxes[face_id] = wp.dot(v,normal)*area

for key,(cell_struct,face_struct,node_struct) in CELL_DICT.items():
    float_type = wp.float32 if 'F32' in key else wp.float64
    wp.overload(calculate_mass_flux_kernel,{"mass_fluxes":wp.array(dtype=float_type),
                                        "face_values": wp.array2d(dtype = float_type),
                                        "cell_structs":wp.array(dtype=cell_struct),
                                        "face_structs":wp.array(dtype=face_struct)})



@wp.kernel
def calculate_gradients_kernel(face_values:wp.array2d(dtype=Any),
                                cell_gradients:wp.array2d(dtype = Any),
                                cell_structs:wp.array(dtype = Any),
                                face_structs:wp.array(dtype = Any),
                                node_structs:wp.array(dtype= Any),
                                output_indices:wp.array(dtype=wp.int32),
                                ):
    #Lets Use Gauss Linear from openFoam for now
    i,face_idx,output_idx = wp.tid() #C, faces_per_cell, num_outputs
    output = output_indices[output_idx]

    face_id = cell_structs[i].faces[face_idx]

    area = face_structs[face_id].area
    normal = cell_structs[i].face_normal[face_idx]
    volume = cell_structs[i].volume
    vec = face_values[face_id,output]*area*normal/volume
    # wp.printf('%f %f %f\n',vec[0],vec[1],vec[2])
    wp.atomic_add(cell_gradients,i,output,vec)

for key,(cell_struct,face_struct,node_struct) in CELL_DICT.items():
    float_type = wp.float32 if 'F32' in key else wp.float64
    wp.overload(calculate_gradients_kernel,{"cell_gradients":wp.array2d(dtype = vector(3,dtype=float_type)),
                                        "face_values": wp.array2d(dtype = float_type),
                                        "cell_structs":wp.array(dtype=cell_struct),
                                        "face_structs":wp.array(dtype=face_struct),
                                        "node_structs":wp.array(dtype= node_struct),
                                        "output_indices":wp.array(dtype=wp.int32),})

