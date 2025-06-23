import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp import vec3
from warp.types import vector
# from src.preprocess import Mesh
import warp.sparse as sparse
if __name__ == '__main__':
    wp.init()
    @wp.kernel
    def test_kernel(a:wp.array(dtype=wp.mat((4,4),dtype=float)  ) ):
        i = wp.tid()
        row = wp.vector(22.,22.,33.,33.,dtype=wp.float32)
        m = wp.mat44(a[i])
        m[0] = row
        (a[i])[0,1] = 1.
        # wp.matrix
        # wp.printf('%f \n',a[i][0,0]
        a[i] = m
        print(a[i])
    m_type = wp.mat((4,4),dtype=float)

    m = m_type()
    arr= wp.array(data=[m],dtype= m_type)
    # b = m_type([[0.,1.],[2.,3.]])
    # print(m[0])
    a = vec3(2.)
    a._length_
    print(arr.dtype)
    print(arr.shape)
    
    wp.launch(test_kernel,dim = 1, inputs= [arr])
   