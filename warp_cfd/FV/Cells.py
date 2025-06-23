from warp.types import vector
from warp import mat
import warp as wp

def _create_cell_struct(nodes_per_cell:int,faces_per_cell:int,num_outputs = 4,dimension:int = 3, float_dtype = wp.float32,int_dtype = wp.int32):
    
    @wp.struct
    class Cell:
        '''
        Stores predefined information such as neighbors, nodes, face normals etc into a convienent struct
        '''
        id:int_dtype
        num_nodes:int_dtype 
        num_faces:int_dtype
        '''(N,) array'''
        centroid:vector(length =  dimension,dtype=float_dtype)
        '''(D,) Vector'''
        volume:float_dtype
        '''Float'''
        nodes: vector(length =  nodes_per_cell,dtype=int_dtype)
        '''(N,) Vector of nodes'''
        faces: vector(length=  faces_per_cell,dtype=int_dtype)
        '''(F,) Vector of Face IDs integer'''
        face_normal: mat(shape = (faces_per_cell,dimension),dtype= float_dtype)
        '''(F,D) Matrix of D vector'''
        face_area: vector(length = faces_per_cell,dtype=float_dtype)
        neighbors: vector(length = faces_per_cell,dtype=int_dtype) # Use -1 for no neighbor
        '''(F,) vector'''
        cell_centroid_to_face_centroid: mat(shape = (faces_per_cell,dimension),dtype=float_dtype)
        '''(F,D) matrix with D Vector containg the distance from the cell centroid to the corresponding face centroid'''
        
        mass_fluxes: vector(length=faces_per_cell,dtype = float_dtype)
        ''' Store the flux dot(u,n)*A'''
        face_sides: vector(length=faces_per_cell,dtype = int_dtype)
        '''(F) Vector indicating for the face which side of the face it is on (0 or 1 side)'''

        offset: int_dtype
        '''Offset for nnz COO array'''
        nnz: int_dtype
        '''Number of non zero coeff (Num neighbors + 1) for each velocity component'''
        face_offset_index:vector(length=faces_per_cell,dtype= int_dtype)
        '''vector indicating the offset index for a given face. 0 if boundary'''

        values: vector(length=num_outputs,dtype = float_dtype)
        gradients: mat(shape = (num_outputs,dimension),dtype= float_dtype)
        value_is_fixed:vector(length=num_outputs,dtype = wp.uint8)
    return Cell


def _create_face_struct(nodes_per_face:int,num_outputs = 4,dimension:int = 3, float_dtype = wp.float32,int_dtype = wp.int32):
    '''
    '''
    @wp.struct
    class Face:
        id: int_dtype
        area: float_dtype
        centroid:vector(length=dimension,dtype= float_dtype)
        nodes: vector(length = nodes_per_face, dtype = int_dtype)
        adjacent_cells: vector(length=2,dtype=int_dtype) #-1 meaning no id (i.e. boundary)
        cell_face_index: vector(length=2,dtype=int_dtype)
        norm_distance: vector(length= 2, dtype= float_dtype) # normalised distance from adjacent cell centroid to face centroid
        area:float_dtype
        cell_distance: vector(length = dimension,dtype= float_dtype) # Distance vector from owner cell centroid 0 to neighbor cell centroid 1
        values: vector(length=num_outputs,dtype = float_dtype)
        gradients: vector(length=num_outputs,dtype = float_dtype)  # Assume only Normal component exists so we only need a vector
        is_boundary: wp.uint8
        value_is_fixed: vector(length= num_outputs,dtype= wp.uint8) # 1 is fixed, 0 is free
        gradient_is_fixed: vector(length= num_outputs,dtype= wp.uint8) 


    return Face


def _create_node_struct(num_outputs = 4,dimension:int = 3, float_dtype = wp.float32,int_dtype = wp.int32):
    @wp.struct
    class Node:
        id: int_dtype
        coordinates: vector(length=dimension,dtype= float_dtype)
        values:vector(length=num_outputs,dtype = float_dtype)
        grads: mat(shape= (num_outputs,dimension),dtype = float_dtype )
    return Node
    

NODE3D = _create_node_struct(dimension=3)

HEX = _create_cell_struct(nodes_per_cell=8,faces_per_cell=6)
HEX_FACE = _create_face_struct(nodes_per_face=4)

TETRA = _create_cell_struct(nodes_per_cell=4,faces_per_cell=3)
TETRA_FACE = _create_face_struct(nodes_per_face=3)


def create_mesh_structs(nodes_per_cell:int,faces_per_cell:int,nodes_per_face:int,num_outputs = 4,dimension:int = 3, float_dtype = wp.float32,int_dtype = wp.int32):
    assert float_dtype == wp.float32 and int_dtype == wp.int32, 'only 32bit int and float are supported'

    if nodes_per_cell == 8 and faces_per_cell == 8 and num_outputs == 4 and dimension == 3 and nodes_per_face == 4:
        return HEX,HEX_FACE,NODE3D
    elif nodes_per_cell == 4 and faces_per_cell == 3 and num_outputs == 4 and dimension ==3 and nodes_per_face == 3:
        return TETRA,TETRA_FACE,NODE3D
    
    else:
        Cell = _create_cell_struct(nodes_per_cell,faces_per_cell,num_outputs,dimension, float_dtype,int_dtype)
        Face = _create_face_struct(nodes_per_face,num_outputs, dimension,float_dtype,int_dtype)
        Node = _create_node_struct(num_outputs,dimension,float_dtype,int_dtype)
        return Cell,Face,Node
    



def init_structs(cells:wp.array,faces:wp.array,nodes:wp.array,cell_properties,face_properties,node_properties:wp.array):
        
        cell_struct = cells.dtype
        face_struct = faces.dtype
        node_struct= nodes.dtype
        num_cells = cells.shape[0]
        num_faces = faces.shape[0]
        num_nodes = nodes.shape[0]
        @wp.kernel
        def init_cell_structs(cell_structs:wp.array(dtype = cell_struct),
                            centroids:wp.array(dtype=cell_properties.centroids.dtype),
                            volumes:wp.array(dtype=cell_properties.volumes.dtype),
                            face_ids:wp.array(dtype = cell_properties.faces.dtype),
                            face_normals:wp.array(dtype = cell_properties.normal.dtype),
                            neighbors:wp.array(dtype = cell_properties.neighbors.dtype),
                            cell_centroid_to_face_centroid:wp.array(dtype = cell_properties.cell_centroid_to_face_centroid.dtype),
                            nodes:wp.array(dtype=cell_properties.nodes.dtype),
                            face_sides:wp.array(dtype=cell_properties.face_side.dtype),
                            nnz_per_cell:wp.array(dtype=cell_properties.nnz_per_cell.dtype),
                            face_offset_index:wp.array(dtype=cell_properties.face_offset_index.dtype),
                            value_is_fixed:wp.array(dtype=cell_properties.value_is_fixed.dtype),
                            face_areas:wp.array(dtype=cell_properties.area.dtype)
                            ):
            i = wp.tid()
            cell_structs[i].id = i
            cell_structs[i].centroid = centroids[i]
            cell_structs[i].volume = volumes[i]
            cell_structs[i].faces = face_ids[i]
            cell_structs[i].face_area = face_areas[i]
            cell_structs[i].face_normal = face_normals[i]
            cell_structs[i].neighbors = neighbors[i]
            cell_structs[i].cell_centroid_to_face_centroid = cell_centroid_to_face_centroid[i]
            cell_structs[i].nodes = nodes[i]
            cell_structs[i].face_sides = face_sides[i]
            cell_structs[i].nnz = nnz_per_cell[i]
            cell_structs[i].face_offset_index = face_offset_index[i]
            cell_structs[i].value_is_fixed = value_is_fixed[i]

        @wp.kernel
        def init_face_structs(face_structs:wp.array(dtype= face_struct),
                            is_boundary_face:wp.array(dtype=face_properties.is_boundary.dtype),
                            boundary_is_fixed:wp.array(dtype=face_properties.boundary_value_is_fixed.dtype),
                            gradient_is_fixed:wp.array(dtype=face_properties.gradient_value_is_fixed.dtype),
                            adjacent_cells:wp.array(dtype=face_properties.adjacent_cells.dtype),
                            centroid:wp.array(dtype=face_properties.centroid.dtype),
                            nodes: wp.array(dtype=face_properties.unique_faces.dtype),
                            cell_structs:wp.array(dtype = cell_struct),
                            cell_face_index:wp.array(dtype=face_properties.cell_face_index.dtype),
                            cell_distance:wp.array(dtype=face_properties.distance.dtype),
                            face_area:wp.array(dtype=cell_properties.area.dtype)
                    ):
            
            i = wp.tid() # Looping throuhg uniques faces
            
            face_structs[i].id = i
            face_structs[i].is_boundary = is_boundary_face[i]
            face_structs[i].value_is_fixed = boundary_is_fixed[i]
            face_structs[i].gradient_is_fixed = gradient_is_fixed[i]
            face_structs[i].centroid = centroid[i]
            face_structs[i].nodes = nodes[i]


            face_structs[i].adjacent_cells = adjacent_cells[i]
            face_structs[i].cell_face_index = cell_face_index[i] # WE just need the area of either cell
            face_structs[i].area = face_area[adjacent_cells[i][0]][cell_face_index[i][0]]

            if is_boundary_face[i] == 0:
                owner_cell = cell_structs[adjacent_cells[i][0]]
                neighbor_cell = cell_structs[adjacent_cells[i][1]]
                owner_centroid_to_face = owner_cell.cell_centroid_to_face_centroid[cell_face_index[i][0]]    
                psi = wp.length(owner_centroid_to_face)/wp.length(wp.sub(owner_cell.centroid,neighbor_cell.centroid))
                
                face_structs[i].norm_distance[0] = psi
                face_structs[i].norm_distance[1] =1.-psi
                face_structs[i].cell_distance = cell_distance[i]
        
        
        @wp.kernel
        def init_node_structs(node_structs:wp.array(dtype=node_struct),
                            coordinates:wp.array(dtype= node_properties.dtype)):
            
            i = wp.tid()
            node_structs[i].id = i
            node_structs[i].coordinates = coordinates[i]


        wp.launch(kernel = init_cell_structs,dim = num_cells,inputs = [cells,
                                                                                        cell_properties.centroids,
                                                                                        cell_properties.volumes,
                                                                                        cell_properties.faces,
                                                                                        cell_properties.normal,
                                                                                        cell_properties.neighbors,
                                                                                        cell_properties.cell_centroid_to_face_centroid,
                                                                                        cell_properties.nodes,
                                                                                        cell_properties.face_side,
                                                                                        cell_properties.nnz_per_cell,
                                                                                        cell_properties.face_offset_index,
                                                                                        cell_properties.value_is_fixed,
                                                                                        cell_properties.area])
        
        wp.launch(kernel=init_face_structs,dim = num_faces,inputs = [
            faces,
            face_properties.is_boundary,
            face_properties.boundary_value_is_fixed,
            face_properties.gradient_value_is_fixed,
            face_properties.adjacent_cells,
            face_properties.centroid,
            face_properties.unique_faces,
            cells,
            face_properties.cell_face_index,
            face_properties.distance,
            cell_properties.area
        ])
        
        wp.launch(kernel=init_node_structs,dim = num_nodes,inputs = [
            nodes,
            node_properties,
        ])

