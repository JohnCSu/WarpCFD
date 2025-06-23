import warp as wp
from warp_cfd.preprocess.mesh import cells_data,faces_data
from warp.types import vector
class Ops:
    def __init__(self,cell_struct,face_struct,node_struct,weight_struct,cell_properties:cells_data,face_properties:faces_data,num_outputs:int,float_dtype = wp.float32,int_dtype = wp.int32):
        self.cell_struct = cell_struct
        self.face_struct = face_struct
        self.node_struct = node_struct
        self.weight_struct = weight_struct

        self.cell_properties = cell_properties 
        self.face_properties = face_properties
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        
        self.num_outputs = num_outputs
        self.num_cells = cell_properties.num_cells
        self.faces_per_cell = cell_properties.faces_per_cell
        self.num_faces = face_properties.num_faces
        self.dimension= cell_properties.dimension
        self.vector_type = vector(self.dimension,self.float_dtype)
        assert self.dimension == 3