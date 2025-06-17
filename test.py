import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp import vec3
from warp.types import vector
from src.preprocess import Mesh
import warp.sparse as sparse
if __name__ == '__main__':
    wp.init()
    # mesh = pv.read('1_0.vtm')['internal']

    # mesh = mesh.extract_cells([0,1,2,3,4,5])
    # m = Mesh(mesh)
    # m.set_boundary_value(0,u = 1,v = 2,w = 3.3,p = 2)
    # x = m.faces.to_NVD_warp()

    # print(x.normal.shape,x.normal.numpy())
    ar = np.eye(12,12,dtype = np.float32)
    ar = ar + np.eye(12,12,k = 2,dtype = np.float32)*2.
    ar = ar + np.eye(12,12,k = 2,dtype = np.float32)*3.
    print(ar)

    rows, cols = np.nonzero(ar)
    row,col,value = wp.array(rows.astype(np.int32)),wp.array(cols.astype(np.int32)),wp.array(ar[rows,cols])
    print(row,col)

    warp_sparse = sparse.bsr_from_triplets(*ar.shape,row,col,value)
    print(warp_sparse)
    
    A = sparse.bsr_get_diag(warp_sparse)


    print(A)
