import warp as wp
from typing import Any



@wp.kernel
def _calculate_rUA_kernel(D_cell:wp.array(dtype=Any),
                                    Ap:wp.array(dtype=Any),
                                    cell_structs:wp.array(dtype=Any)
                                    ):
    i = wp.tid() # By Cells
    # diagonals for a cell in u,v,w are always the same for isotropic viscosity
    D = 3
    row = i*D
    D_cell[i] = cell_structs[i].volume/Ap[row]




@wp.kernel
def _interpolate_cell_value_to_face_kernel(D_face:wp.array(dtype=Any),
                                    D_cell:wp.array(dtype=Any),
                                    face_structs:wp.array(dtype = Any)):
    face_id = wp.tid() # K
    # D is a C,3 array This is always 3
    owner_cell_id = face_structs[face_id].adjacent_cells[0]
    owner_D = D_cell[owner_cell_id]
    if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
        D_face[face_id] = owner_D
    else:
        neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
        neighbor_D = D_cell[neighbor_cell_id]
        
        D_f = face_structs[face_id].norm_distance[0]*owner_D + face_structs[face_id].norm_distance[1]*neighbor_D
        D_face[face_id] = D_f




def calculate_rUA(Ap:wp.array,cells:wp.array,out:wp.array | None = None):

    '''
    Calculate rUA =  1/ap = V/Ap where Ap is the diagonal coeffecient of the momentum matrix
    '''
    
    if out is None:
        out = wp.zeros_like(Ap)
    else:
        assert out.shape[0]*3 == Ap.shape[0], 'Ap is diagonal of 3 velocities so should be 3*num_cells in model'
    
    assert len(Ap.shape) == 1
    
    wp.launch(_calculate_rUA_kernel,dim = cells.shape[0],inputs= [out,Ap,cells])
    return out


def interpolate_cell_value_to_face(face_value:wp.array,cell_value:wp.array,faces:wp.array):
    wp.launch(_interpolate_cell_value_to_face_kernel,dim = faces.shape[0],inputs= [face_value,cell_value,faces])
    return face_value