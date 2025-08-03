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
        



def mult_by_volume(arr:wp.array,cells:wp.array):
    wp.launch(kernel=mult_by_volume_kernel,dim = cells.shape[0],inputs=[arr,cells])

@wp.kernel
def mult_by_volume_kernel(arr:wp.array(dtype=float),cell_structs:wp.array(dtype=Any)):
    i = wp.tid()

    arr[i] = arr[i]*cell_structs[i].volume



def to_vector_array(array:np.ndarray,float_dtype = wp.float32):
    '''
    Convert numpy array to warp array with dtype = vector. For shapes >=2, the last array axis is treated 
    as the vector dimension so a (N,M) numpy array becomes an (N,) warp array with dtype: vector(length=M)
    '''
    # dtype = wp.dtype_from_numpy(array.dtype)
    shape = array.shape
    n = shape[-1]
    vector_type = wp.types.vector(length = n, dtype = float_dtype)
    return wp.array(data = array,dtype=vector_type)