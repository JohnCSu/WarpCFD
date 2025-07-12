import warp as wp
import numpy as np
import warp.sparse as sparse
import scipy.sparse as sp_sparse
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
