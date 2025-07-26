import warp as wp
from typing import Any
def central_difference(float_dtype):
    @wp.func
    def _central_difference(
                        cell_values:wp.array2d(dtype = float_dtype),
                        cell_gradients:wp.array2d(dtype = float_dtype),
                        mass_fluxes:wp.array(dtype = float_dtype),
                        owner_cell:Any,
                            neighbor_cell:Any,
                            face:Any,
                            output:int,
                            ):

        psi = face.norm_distance[0]
        psi_2 = face.norm_distance[1] # Equal to (1-psi)
        weight_vec = wp.vector(length=3,dtype=float_dtype)
        #Owner Weight
        weight_vec[0] = psi
        #Neighbor Weight
        weight_vec[1] = psi_2
        #Explicit Weight
        weight_vec[2] = float_dtype(0.)

        return weight_vec
    return _central_difference

def upwind(float_dtype):
    @wp.func
    def _upwind( 
                cell_values:wp.array2d(dtype = float_dtype),
                cell_gradients:wp.array2d(dtype = Any),
                mass_fluxes:wp.array(dtype = float_dtype),
                owner_cell:Any,
                neighbor_cell:Any,
                face:Any,
                output:int,
                ):
        # 0 -> Owner, 1 -> Neighbor, 2 -> Explicit RHS term
        weight_vec = wp.vector(length=3,dtype=float_dtype)

        if mass_fluxes[face.id] > float_dtype(0.):
            weight_vec[0]= float_dtype(1.)
            weight_vec[1] = float_dtype(0.)
        else:
            weight_vec[0]= float_dtype(0.)
            weight_vec[1]= float_dtype(1.)
        
        weight_vec[2] = float_dtype(0.)

        return weight_vec
    return _upwind

def upwindLinear(float_dtype):
    @wp.func
    def _upwindLinear( 
                cell_values:wp.array2d(dtype = float_dtype),
                cell_gradients:wp.array2d(dtype = Any),
                mass_fluxes:wp.array(dtype = float_dtype),
                owner_cell:Any,
                neighbor_cell:Any,
                face:Any,
                output:int,
                ):
        '''
        Upwind Linear 
        '''

        # 0 -> Owner, 1 -> Neighbor, 2 -> Explicit RHS term
        weight_vec = wp.vector(length=3,dtype=float_dtype)

        j = face.cell_face_index

        if mass_fluxes[face.id] > float_dtype(0.):
            weight_vec[0]= float_dtype(1.)
            weight_vec[1] = float_dtype(0.)
            upwind_id = owner_cell.id
            dist_vec = owner_cell.cell_centroid_to_face_centroid[j[0]]
        else:
            weight_vec[0]= float_dtype(0.)
            weight_vec[1]= float_dtype(1.)
            upwind_id = neighbor_cell.id
            dist_vec = owner_cell.cell_centroid_to_face_centroid[j[1]]

        weight_vec[2] = wp.dot(cell_gradients[upwind_id,output],dist_vec)
        

        return weight_vec
    return _upwindLinear