import warp as wp
from typing import Any
'''
wp.functions for laplacian schemes.

There are many ways to do the correction. Currently implemented is the Over relaxed approach given
in this approach https://www.youtube.com/watch?v=yU7r8mYK3bs&t=1s

Note that for these functions you only need to calculate the grad(y) dot n terms, outside,
the function are multiplied by the area and viscosity

'''

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
def central_difference_corrected(float_dtype):
    @wp.func
    def _central_difference_corrected( cell_values:wp.array2d(dtype = float_dtype),
                            cell_gradients:wp.array2d(dtype = Any),
                            mass_fluxes:wp.array(dtype = float_dtype),
                            owner_cell:Any,
                                neighbor_cell:Any,
                                face:Any,
                                global_output_idx:int,):
        
        n_f = owner_cell.face_normal[face.cell_face_index[0]]
        d = face.cell_distance
        d_mag = wp.length(d)
        # Get n1 along d direction, n_f is normalised
        n1 = d*(float_dtype(1.)/wp.dot(n_f,d))
        n1_mag = wp.length(n1)
        #Orthogonal
        n2 = n_f-n1

        weight_vec = wp.vector(length=3,dtype=float_dtype)
        #Owner Weight
        weight_vec[0] = float_dtype(-1.)/d_mag*n1_mag
        #Neighbor Weight
        weight_vec[1] = float_dtype(1.)/d_mag*n1_mag
        #Explicit Weight Need interp gradients
        grad_face = face.norm_distance[0]*cell_gradients[owner_cell.id,global_output_idx] + face.norm_distance[1]*cell_gradients[neighbor_cell.id,global_output_idx]
        weight_vec[2] = wp.dot(grad_face,n2)

        return weight_vec
    return _central_difference_corrected