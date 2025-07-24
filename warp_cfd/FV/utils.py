import warp as wp
import numpy as np
import warp.sparse as sparse
import scipy.sparse as sp_sparse
from typing import Any
class COO_Arrays:
    rows: wp.array
    cols: wp.array
    values: wp.array
    nnz: int
    offsets: wp.array

    def __init__(self,nnz_per_cell: wp.array,num_outputs,float_dtype,int_dtype):
        # Need to include cell itself

        nnz_per_cell:np.ndarray = nnz_per_cell.numpy()

        num_cells =nnz_per_cell.shape[0]
        nnz = int(num_outputs*nnz_per_cell.sum())
        offsets = np.zeros_like(nnz_per_cell)
        offsets[1:] = np.cumsum(num_outputs*nnz_per_cell)[:-1]        
        self.nnz = nnz

        self.offsets = wp.array(data = offsets)
        self.rows = wp.zeros(nnz,dtype=int_dtype)
        self.cols = wp.zeros(nnz,dtype=int_dtype)
        self.values = wp.zeros(nnz,dtype=float_dtype)


def bsr_to_coo_array(bsr:sparse.BsrMatrix):
    return sp_sparse.coo_array((bsr.values.numpy(),(bsr.uncompress_rows().numpy(),bsr.columns.numpy())),shape = (bsr.nrow,bsr.ncol))


def extrapolate_to_faces(self,D_face,cells,faces,weights):
    faces_per_cell = cells.vars['faces'].type._length_
    wp.launch(kernel= self.extrapolate_to_faces,dim = [cells.shape[0],faces_per_cell,1],inputs=[cells,faces,weights,D_face])
        


@wp.kernel
def extrapolate_to_faces(face_values:wp.array(dtype=float),
                        cell_values:wp.array(dtype=float),
                        face_structs:wp.array(dtype = Any)):
            face_id = wp.tid() # K
            # D is a C,3 array This is always 3
            owner_cell_id = face_structs[face_id].adjacent_cells[0]
            owner_D = cell_values[owner_cell_id]
            if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
                face_values[face_id] = owner_D
            else:
                neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
                neighbor_D = cell_values[neighbor_cell_id]
                
                D_f = face_structs[face_id].norm_distance[0]*owner_D + face_structs[face_id].norm_distance[1]*neighbor_D
                face_values[face_id] = D_f



def mult_by_volume(arr:wp.array,cells:wp.array):
    wp.launch(kernel=mult_by_volume_kernel,dim = cells.shape[0],inputs=[arr,cells])

@wp.kernel
def mult_by_volume_kernel(arr:wp.array(dtype=float),cell_structs:wp.array(dtype=Any)):
    i = wp.tid()

    arr[i] = arr[i]*cell_structs[i].volume




def green_gauss_gradient(val_arr:wp.array,grad_arr:wp.array,coeff: float | wp.array,fv):
    if isinstance(coeff,float):
        coeff = wp.array([coeff],dtype=val_arr.dtype)
    assert coeff.shape[0] == 1 or coeff.shape[0] == fv.num_faces
    grad_arr.zero_()
    wp.launch(kernel = green_gauss_gradient_kernel,dim = (fv.num_cells,fv.faces_per_cell),inputs = [val_arr,grad_arr,fv.cells,fv.faces,coeff])
    return grad_arr


@wp.kernel
def green_gauss_gradient_kernel(val_arr:wp.array(dtype = Any),
                         grad_arr:wp.array2d(dtype = Any),
                         cell_structs:wp.array(dtype=Any),
                         face_structs:wp.array(dtype=Any),
                         coeff:wp.array(dtype=Any)
                         ):
    i,j = wp.tid() # C,F


    
    current_cell = cell_structs[i]
    neighbor_cell_id = current_cell.neighbors[j]

    face_id = cell_structs[i].faces[j]
    if coeff.shape[0] == 1:
        coeff_ = coeff[0]
    else:
        coeff_ = coeff[face_id]
    
    face = face_structs[face_id]
    
    if face.is_boundary:
        face_value = val_arr[current_cell.id]
    else:
        if current_cell.face_sides[j] == 0: # Current Cell is owner side of said face
            face_value = face.norm_distance[0]*val_arr[i] + face.norm_distance[1]*val_arr[neighbor_cell_id]
        else:
            face_value = face.norm_distance[1]*val_arr[i] + face.norm_distance[0]*val_arr[neighbor_cell_id]

    area = face_structs[face_id].area
    normal = current_cell.face_normal[j]
    volume = current_cell.volume

    vec = coeff_*face_value*area*normal/volume

    for k in range(3):
        wp.atomic_add(grad_arr,i,k,vec[k]) 
