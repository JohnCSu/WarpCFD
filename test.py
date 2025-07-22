from warp_cfd.preprocess.grid import create_2D_grid
from warp_cfd.preprocess.mesh import Mesh

pv_mesh = create_2D_grid((0,0,0),1,1,1,1,element_type= 'hex',unstructured_wedge = False,display_mesh=False)
Mesh(pv_mesh)