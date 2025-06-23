import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType



class group():
    name:str
    ids: np.ndarray
    type: str
    def __init__(self,name,ids,type) -> None:
        self.name = name
        self.ids = ids
        self.type = type

class Mesh():
    def __init__(self,pyvista_mesh: pv.UnstructuredGrid |pv.StructuredGrid,dtype = np.float32,num_outputs = 4) -> None:
        assert isinstance(pyvista_mesh,(pv.UnstructuredGrid,pv.StructuredGrid))
        assert len(list(pyvista_mesh.cells_dict.keys())) == 1, 'Meshes can only contain a single cell type, Got multiple'
        
        self.pyvista_mesh = pyvista_mesh
        self.nodes = np.array(pyvista_mesh.points)
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        _celltype = list(pyvista_mesh.cells_dict.keys())[0]
        self.dtype = dtype
        self.cellType = (CellType(_celltype).name,_celltype)
        self.cells = np.array(pyvista_mesh.cells_dict[self.cellType[1]],dtype=self.int_dtype) 
        self.cell_centroids = np.array(pyvista_mesh.cell_centers().points,dtype=self.float_dtype)
        self.cell_volumes = np.array(pyvista_mesh.compute_cell_sizes()['Volume'],dtype=self.float_dtype)
        self.groups:dict[str,group] = {}
        self.cell_check()
        self.num_outputs = num_outputs
        
        self.face_properties,self.cell_properties = self.get_mesh_properties()
        
        
        self.gridType = 'Unstructured' if isinstance(pyvista_mesh,pv.UnstructuredGrid) else 'Structured'
    @property
    def dimension(self):
        assert self.nodes.shape[-1] == 2 or self.nodes.shape[-1] == 3 
        return self.nodes.shape[-1]


    def cell_check(self):
        if self.dimension == 3:
            assert self.cellType[1] == CellType.HEXAHEDRON or self.cellType[1] == CellType.TETRA, 'Currently only Tetra and Hexahedron elements are supported' 

    def add_group(self,name,id,group_type):
        group_types = {'face','cell','point','edge'}
        assert group_type in group_types, 'group_type must be one of the following: {face,cell,point,edge}'
        assert name not in self.groups.keys(), 'name for group already exists!'
        self.groups[name] = group(name,id,group_type)


    def add_groups(self,dic:dict,group_type):
        for name,id in dic.items():
            self.add_group(name,id,group_type)

    def get_faces(self):
        if self.dimension == 2:
            #In 2D Faces are edges
            self.faces = self.pyvista_mesh.extract_all_edges(clear_data=True).lines
        elif self.dimension == 3:
            hex_faces = np.array([
                [3, 2, 1, 0],  # Bottom face
                [4, 5, 6, 7],  # Top face
                [0, 1, 5, 4],  # Front face
                [1, 2, 6, 5],  # Right face
                [3, 7, 6, 2],  # Back face
                [3, 0, 4, 7]   # Left face
            ],dtype=self.int_dtype)

            tet_faces = np.array([
                [2, 1, 0],
                [0, 1, 3],
                [3, 2, 0],
                [1, 2, 3]
            ],dtype=self.int_dtype)



            face_idx = tet_faces if self.cellType[1] == CellType.TETRA else hex_faces
            # self.face_idx = face_idx

            faces = self.cells[:,face_idx] # (C,F,N) C number of cells, F number of faces per cell, N number of nodes per cell
            faces_coords = self.nodes[faces] # (C,F,N,3)
            faces_sorted = np.sort(faces, axis=2)

            # Step 3: Reshape the array to have shape (num_cells*4, 3), where each row is a face.
            faces_reshaped = faces_sorted.reshape(-1, face_idx.shape[-1])

            # Step 4: Use np.unique to get the unique faces along axis=0.
            unique_faces, face_ids = np.unique(faces_reshaped, axis=0, return_inverse=True)

            face_ids = face_ids.reshape(self.cells.shape[0], face_idx.shape[0])
            face_ids = face_ids.astype(dtype=self.int_dtype) # Watch this line carefully
            return faces,faces_coords,unique_faces,face_ids

    @staticmethod
    def calculate_face_normal_and_area(faces,nodes):

        faces_coords = nodes[faces]
        # Anchor
        if faces.shape[-1] < 3:
            raise ValueError('The number of nodes defined for each face must be at least 3 or more nodes')
        # (C,F,N,3)
        v0 = faces_coords[:,:,0,:] # C,F,3 Array
        
        areas = []
        normal_vectors = []
        for i in range(1,faces.shape[-1]-1):
            vi = faces_coords[:,:,i,:]
            vii = faces_coords[:,:,i+1,:]
            normal_vector = 1/2*np.cross((vi - v0),(vii-v0)) # Default uses last axis
            area =  np.linalg.norm(normal_vector,axis = -1)
            areas.append(area)
            normal_vectors.append(normal_vector)
        face_areas = sum(areas)
        face_normals = sum(normal_vectors)/face_areas[:,:,np.newaxis]

        return face_areas,face_normals
    
    
    def get_neighbors(self) -> tuple[np.ndarray]:
        '''
        Return all the neighbors of the centroids of each cell. Assumes that all cells are connected to at least another cell



        Returns: `(D,2) np.array` where D is the number of unique neighbors describing face connectivity in the mesh.
          the 3 elements in each row are `(cell_1,cell_2,dist)` where cell 1 and cell 2 are connected cells and 
          dist is the euclidean distance between the centroids of the 2 cells

          The edge (cell_1,cell_2) is undirected and is sorted such that the lower cell id is always the first element 
        '''

        connections = 'faces' if self.dimension == 3 else 'edges'
        cell_neighbors = np.array([ [i,j] for i in range(len(self.cells)) for j in self.pyvista_mesh.cell_neighbors(i,connections=connections)],dtype = self.int_dtype)
        
        # We want Unique Pairs
        
        return np.unique(np.sort(cell_neighbors,axis= -1),axis = 0)
    

    def get_mesh_properties(self):
        cell_neighbors = self.get_neighbors()
        faces,faces_coords,unique_faces,cell_face_ids = self.get_faces()

        face_areas,face_normals = self.calculate_face_normal_and_area(faces,self.nodes)
        # Face Areas and normals is (C,F) and (C,F,D)
        # Cell Neightbors is (C,k_i) list (variable)

        #We should extend the list to also have: Face ID, Face Normal, Face Area, Cell-connected_to, Distance
        C,F,N = faces.shape
        K = unique_faces.shape[0]
        D = self.dimension
        O = self.num_outputs
        faces = faces_data()
        cells = cells_data()
        #We can populate all FaceID, Face Normal And Face Area first
        cells.num_cells=C
        cells.faces_per_cell = F
        cells.nodes_per_face = N
        faces.num_faces = K
        

        faces.unique_faces = unique_faces # (K,N)
        faces.dimension = self.dimension
        faces.centroid = self.nodes[faces.unique_faces].mean(axis=1)
        faces.adjacent_cells = np.ones((K,2),dtype = self.int_dtype)*(-1) # -1 means no neighbors
        faces.distance= np.ones((K,self.dimension),dtype= self.float_dtype)*(-1)
        faces.is_boundary = np.ones((K,),dtype=np.uint8) # (K,) bool 1 if BC 0 otherwise for each face
        faces.boundary_value_is_fixed = np.zeros((K,O),dtype=np.uint8) 
        faces.boundary_value = np.zeros((K,O),dtype= self.float_dtype) # (K, dim+1) dim+1 number of output variables (u,v,w,p)
        faces.gradient_value_is_fixed = np.zeros((K,O),dtype=np.uint8)
        faces.gradient_value_is_fixed[:,3] = 1 # P is output no 3 and we assume by default all is non-slip walls

        faces.gradient_value = np.zeros((K,O),dtype= self.float_dtype) # (K, dim+1) dim+1 number of output variables (u,v,w,p)
        faces.cell_face_index = np.ones((K,2),dtype = self.int_dtype)*(-1)

        cells.dimension = self.dimension
        cells.centroids = self.cell_centroids
        cells.volumes = self.cell_volumes
        cells.nodes = self.cells
        cells.face_coords = faces_coords # (C,F,N,D)
        cells.faces = cell_face_ids
        cells.normal = face_normals
        cells.area = face_areas
        cells.cell_centroid_to_face_centroid =  faces.centroid[cells.faces] - self.cell_centroids[:,np.newaxis,:]
        cells.neighbors = np.ones((C,F),dtype = self.int_dtype)*(-1) # --1 for no neighbot (external face), otherwise give cell neighbor ID and faceID
        cells.face_side = np.zeros((C,F),dtype = self.int_dtype)
        cells.nnz_per_cell = np.ones(C,dtype = self.int_dtype) # add 1 each time a cell appears in the for loop below
        cells.value_is_fixed = np.zeros((C,O),dtype= np.uint8) 
        cells.fixed_value = np.zeros((C,O),dtype= self.float_dtype) 
        
        for i,j in cell_neighbors: # Get Cell i and j ID
            faces_i,faces_j = cell_face_ids[i],cell_face_ids[j]
            intersect_face_id,face_i_ind, face_j_ind = np.intersect1d(faces_i,faces_j,return_indices= True)
            
            assert face_i_ind.shape[0] == 1, 'If this is raised then a cell shares a face with multiple cells'
            #Neighbor we need to store: cellID and then the associated face ID
            cells.neighbors[i,face_i_ind[0]] = j            
            cells.neighbors[j,face_j_ind[0]] = i
            cells.face_side[i,face_i_ind[0]] = 0
            cells.face_side[j,face_j_ind[0]] = 1
            
            cells.nnz_per_cell[i] += 1
            cells.nnz_per_cell[j] += 1
            
            faces.distance[intersect_face_id] = self.cell_centroids[j] - self.cell_centroids[i]
            
            faces.is_boundary[intersect_face_id] = 0
            faces.adjacent_cells[intersect_face_id] = [i,j]
            faces.cell_face_index[intersect_face_id] = [face_i_ind[0],face_j_ind[0]] # if no neighbors then boundary face so leave as [-1,-1]
        
        
        # For Boundary Faces set 0 index of adjacent cells to the current cell
        for cell_id,cell_face_idx in np.argwhere(cells.neighbors == -1):
            face_id = cells.faces[cell_id,cell_face_idx]
            faces.adjacent_cells[face_id,0] = cell_id
            faces.cell_face_index[face_id,0] = cell_face_idx

        # get ids for boundary and internal faces
        face_ids = np.arange(K)
        faces.boundary_face_ids = face_ids[faces.is_boundary == 1]
        faces.internal_face_ids = face_ids[faces.is_boundary == 0]

        # Get Offset index Tahnks Chat GPT!
        cells.face_offset_index = (cells.neighbors != -1)
        cells.face_offset_index = np.array(np.cumsum(cells.face_offset_index,axis=1)*cells.face_offset_index,dtype= self.int_dtype)
        
         


        return faces,cells


    def set_boundary_value(self,face_ids:str | int|list|tuple|np.ndarray,u = None,v=None,w=None,p=None,overwrite_gradient = True):

        if isinstance(face_ids,str):
            assert face_ids in list(self.groups.keys()), 'Specified group name does not exist'
            group_face_ids = self.groups[face_ids].ids
            self.face_properties.set_boundary_value(group_face_ids,u,v,w,p,overwrite_gradient)

        elif isinstance(face_ids,(int,list,tuple,np.ndarray)):
            self.face_properties.set_boundary_value(face_ids,u,v,w,p,overwrite_gradient)

        else:
            raise ValueError(f'face_ids can be type string,int,list,tuple, or np.ndarray got {type(face_ids)} instead')

    

    def set_gradient_value(self,face_ids:str | int|list|tuple|np.ndarray,u = None,v=None,w=None,p=None,overwrite_boundary = False):

        if isinstance(face_ids,str):
            assert face_ids in list(self.groups.keys()), 'Specified group name does not exist'
            group_face_ids = self.groups[face_ids].ids
            self.face_properties.set_gradient_value(group_face_ids,u,v,w,p,overwrite_boundary)

        elif isinstance(face_ids,(int,list,tuple,np.ndarray)):
            self.face_properties.set_gradient_value(face_ids,u,v,w,p,overwrite_boundary)

        else:
            raise ValueError(f'face_ids can be type string,int,list,tuple, or np.ndarray got {type(face_ids)} instead')

    def set_cell_value(self,cell_ids:str | int|list|tuple|np.ndarray,u = None,v=None,w=None,p=None):

        if isinstance(cell_ids,str):
            assert cell_ids in list(self.groups.keys()), 'Specified group name does not exist'
            group_cell_ids = self.groups[cell_ids].ids
            self.cell_properties.set_cell_value(group_cell_ids,u,v,w,p)

        elif isinstance(cell_ids,(int,list,tuple,np.ndarray)):
            self.cell_properties.set_cell_value(cell_ids,u,v,w,p)

        else:
            raise ValueError(f'cell_ids can be type string,int,list,tuple, or np.ndarray got {type(cell_ids)} instead')





class cells_data():
    faces: np.ndarray| wp.array
    '''(C,F) array where C is the number of Cells and F is the number of faces per cell. Each element value is an integer corresponding to the face found in `unique_faces`'''
    normal: np.ndarray| wp.array
    '''(C,F,D) array giving the normal of each face for each Cell.'''
    area: np.ndarray| wp.array
    '''(C,F) array giving the area of each face. This should be reduces to K,3 but for convience the first 2 axis are the same shape as `normal`'''
    neighbors: np.ndarray| wp.array
    '''(C,F) array that describes if a face is connected to another Cell, return Cell Neighbor ID.'''
    centroids: np.ndarray| wp.array
    '''C,D array with cell centroid and coordinates'''
    volumes:np.ndarray| wp.array
    '''C array of cell volumes'''
    nodes:np.ndarray| wp.array
    '''C,N array with node ID'''
    cell_centroid_to_face_centroid: np.ndarray| wp.array
    '''(C,F,D) array containg the distance from the cell centroid to the corresponding face centroid'''
    face_side:np.ndarray| wp.array
    '''(C,F) array that describes if the cell is on the 0 side of the face or the 1 side of the face '''
    nnz_per_cell:np.ndarray| wp.array
    '''(C,) array that counts the number of non zero contributions within each cell. This is equavalent to the num neighbors + 1 ( to incl the cell itself) in each cell'''
    face_coords:  np.ndarray | wp.array
    '''(C,F,N,D) array containing the nodal coordinates of each face. For convience it is over all Cells'''

    face_offset_index: np.ndarray | wp.array
    '''(C,F) array  that contains the offset index when forming the COO arrays, 0 if face is a boundary face and has no neighbors'''


    fixed_value: np.ndarray | wp.array
    '''(C,O) the persribed cell centered value. This should be primarly used for setting a reference pressure. Do not use for velocity yet'''
    value_is_fixed: np.ndarray | wp.array
    '''(C,O) whether the cell centered value is fixed or not. This should be primarly used for setting a reference pressure. Do not use for velocity yet'''

    fixed_cells: np.ndarray | wp.array
    '''Array of cell ids that have fixed values'''

    dimension: int
    ''' Dimension of mesh'''
    array_type: str = 'numpy'
    ''' String indicating whether the attributes are `numpy` or `taichi`'''
    float_dtype: np.dtype = np.float32
    '''dtype used for floating point arrays. Default float32'''

    int_dtype: np.dtype = np.int32
    '''dtype used for interger point arrays. Default float32'''
    
    

    num_cells:int
    '''Number of cells'''
    faces_per_cell:int
    '''Number of faces PER cell'''
    nodes_per_face:int
    '''Number of nodes PER FACE'''

    vars_mapping: dict = {
            'u': 0, 
            'v': 1,
            'w': 2,
            'p': 3,
        }


    def set_cell_value(self,cell_ids: int | list |tuple |np.ndarray,u = None,v=None,w=None,p=None):

        assert u is None and v is None and w is None, 'Only pressure can be perscribed to cell value at the moment'

        assert isinstance(cell_ids,(int,list,tuple,np.ndarray))
        
        if not isinstance(cell_ids,(list,tuple,np.ndarray)):
            cell_ids = np.array([cell_ids])

        vars_mapping =self.vars_mapping
        ''' Mapping of variable names to order'''
        values_to_set = {name:val for name,val in zip(vars_mapping.keys(),[u,v,w,p]) if val is not None}
        vars_to_fix = [idx for key,idx in vars_mapping.items() if key in values_to_set.keys() ]
        idx = np.ix_(cell_ids,vars_to_fix)
        self.fixed_value[idx] = list(values_to_set.values())
        self.value_is_fixed[idx] = True



    def to_NVD_warp(self):
        '''Convert Arrays to Warp Arrays, if the last dimension matches that of the dimension in faces class, assume its a vector'''

        
        cells = cells_data()
        cells.array_type = 'warp'

        self.fixed_cells = np.nonzero(self.value_is_fixed[:,3])[0].astype(self.int_dtype) # We only take pressure for now

        for key,val in self.__dict__.items():
            if isinstance(val,np.ndarray):
                # The last array is the vector lengh

                shape = val.shape
                n = val.shape[-1]
                if len(val.shape) == 3 : # Treat this as a matrix
                    dtype = wp.dtype_from_numpy(val.dtype)
                    # vector_type = wp.types.vector(length = n, dtype = dtype)
                    matrix_type = wp.mat(shape = (val.shape[1:]),dtype=dtype)
                    f = wp.array(data = val, dtype = matrix_type )
                elif len(val.shape) == 2:
                    if n == self.dimension: # If the last matches dimension then we set the last axis as a vec3 
                        f = wp.array(data = val,dtype=wp.vec3)
                    else:
                        dtype = wp.dtype_from_numpy(val.dtype)
                        vector_type = wp.types.vector(length = n, dtype = dtype)
                        f = wp.array(data = val, dtype=vector_type)
                else:
                    f = wp.array(data = val) # implicit use the dtype defined in numpy array

                setattr(cells,key,f)
            else:
                setattr(cells,key,val)
        return cells
    


class faces_data():
    '''
    Stores Info on faces:
        shape Dimensions:
            - C - Number of cells in Mesh
            - N - Number of nodes PER face
            - D - Dimension of model (usually 3)
            - K - Number of unique faces in mesh
            - F - Number of faces PER cell

        Keys:
    - `unique_faces` - a (K,N,D) where K is the number of unique faces in the mesh and N is the number of nodes per face. Returns the nodal coordinates of each face    
    - `face coords` a (C,F,N,D) array containing the nodal coordinates of each face. For convience it is over all Cells
    - 'centroid' a (C,F,D) array containing the centroid point of each face
    - `id` - a (C,F) array where C is the number of Cells and F is the number of faces per cell. Each element value is an integer corresponding to the face found in `face coords`
    - `normal` - a (C,F,D) array giving the normal of each face for each Cell.
    - `area` - a (C,F) array giving the area of each face. This should be reduces to K,3 but for convience the first 2 axis are the same shape as `normal`
    - `neighbors` - a (C,F,3) array that describes if a face is connected to another Cell. for a given Cell and Face, the 2 values are (Cell neighbor ID, FaceID). If the face is not connected to any neighbors both values are set to -1
    - `distance` - a (K,D) array that stores the centroid distance between 2 neighboring cells. If a face is not connected to a neighboring cell, distance is set to -1
    - `is_boundary` a (K,) array that stores a boolean on whether a unique face is a boundary/external face (i.e. no neighbors) or internal face
    - `boundary_value` a (K,D+1) specifying the values (u,v,w,p in 3D) given at each face. Default all 0
    '''
    
    unique_faces : np.ndarray | wp.array
    '''(K,N) where K is the number of unique faces in the mesh and N is the number of nodes per face. Returns the nodal coordinates of each face '''
    
    centroid: np.ndarray | wp.array
    '''(K,D) array containing the centroid point of each face'''
    is_boundary: np.ndarray| wp.array
    '''(K,) array that stores a boolean on whether a unique face is a boundary/external face (i.e. no neighbors) or internal face and if that variable at that face is a BC'''
    boundary_value: np.ndarray| wp.array
    ''' (K,D+1) specifying the values (u,v,w,p in 3D) given at each face. Default all 0'''
    gradient_value: np.ndarray| wp.array
    ''' (K,D+1) specifying the  gradient of the variable (u,v,w,p) wrt the normal direction given at each face. Default all 0'''
    boundary_value_is_fixed : np.ndarray| wp.array
    ''' (K,D+1) array specifying if the variable (u,v,w,p) is free or fixed. set to True if value id determined and fixed'''
    gradient_value_is_fixed : np.ndarray| wp.array
    ''' (K,D+1) array specifying if the gradient of the variable (u,v,w,p) wrt the normal direction is free or fixed. set to True if value id determined and fixed'''
    boundary_face_ids : np.ndarray| wp.array
    '''(B) array of faces ids that are boundary faces''' 
    internal_face_ids : np.ndarray| wp.array 
    '''(K-B) array of face ids that are internal faces'''
    adjacent_cells: np.ndarray| wp.array
    '''(K,2) array for a given face, return the adjacent cells'''
    cell_face_index: np.ndarray| wp.array
    '''(K,2) array for a given face ID, return the face index that face ID occupies in the adjacent cells'''

    distance: np.ndarray| wp.array
    '''(K,3) array that stores the centroid distance vector between 2 neighboring cells. If a face is not connected to a neighboring cell, distance is set to -1'''
    
    dimension: int
    ''' Dimension of mesh'''
    array_type: str = 'numpy'
    ''' String indicating whether the attributes are `numpy` or `taichi`'''
    dtype: np.dtype = np.float32
    '''dtype used for floating point arrays. Default float32'''

    int_dtype: np.dtype = np.int32
    '''dtype used for interger point arrays. Default float32'''

    nodes_per_face:int
    '''Number of nodes PER FACE'''
    num_faces:int
    '''Number of Unique Faces'''
    vars_mapping: dict = {
            'u': 0, 
            'v': 1,
            'w': 2,
            'p': 3,
        }

    def set_boundary_value(self,face_ids: int | list |tuple |np.ndarray,u = None,v=None,w=None,p=None,overwrite_gradient= True):
        assert isinstance(face_ids,(int,list,tuple,np.ndarray))
        
        if not isinstance(face_ids,(list,tuple,np.ndarray)):
            face_ids = np.array([face_ids])

        assert np.all(self.is_boundary[face_ids]), 'One of the provided face id is not an external face valid for applying boundary conditions'
        ''' we either have a float for all in face_ids or an array of same len'''

        for i,var in enumerate([u,v,w,p]):
            if var is not None:
                self.boundary_value[face_ids,i] = var
                self.boundary_value_is_fixed[face_ids,i] = True
                if overwrite_gradient:
                    self.gradient_value_is_fixed[face_ids,i] = False       

        

    def set_gradient_value(self,face_ids: int | list |tuple |np.ndarray,u = None,v=None,w=None,p=None,overwrite_boundary = False):
        #Set the gradient value normal to each face
        assert isinstance(face_ids,(int,list,tuple,np.ndarray))
        
        if not isinstance(face_ids,(list,tuple,np.ndarray)):
            face_ids = np.array([face_ids])

        assert np.all(self.is_boundary[face_ids]), 'One of the provided face id is not an external face valid for applying boundary conditions'

        vars_mapping =self.vars_mapping
        ''' Mapping of variable names to order'''
        values_to_set = {name:val for name,val in zip(vars_mapping.keys(),[u,v,w,p]) if val is not None}
        vars_to_fix = [idx for key,idx in vars_mapping.items() if key in values_to_set.keys() ]

        idx = np.ix_(face_ids,vars_to_fix)
        self.gradient_value[idx] = list(values_to_set.values())
        self.gradient_value_is_fixed[idx] = True

        if overwrite_boundary:
            self.boundary_value_is_fixed[idx] = False
        
    def to_NVD_warp(self):
        '''Convert Arrays to Warp Arrays, if the last dimension matches that of the dimension in faces class, assume its a vector'''


        faces = faces_data()
        faces.array_type = 'warp'

        # C = self.id.shape[0]

        for key,val in self.__dict__.items():
            if isinstance(val,np.ndarray):
                # The last array is the vector lengh

                shape = val.shape
                n = val.shape[-1]
                if len(val.shape) == 3 : # Treat this as a matrix
                    dtype = wp.dtype_from_numpy(val.dtype)
                    # vector_type = wp.types.vector(length = n, dtype = dtype)
                    matrix_type = wp.mat(shape = (val.shape[1:]),dtype=dtype)
                    f = wp.array(data = val, dtype = matrix_type )
                elif len(val.shape) == 2:
                    if n == self.dimension: # If the last matches dimension then we set the last axis as a vec3 
                        f = wp.array(data = val,dtype=wp.vec3)
                    else:
                        dtype = wp.dtype_from_numpy(val.dtype)
                        vector_type = wp.types.vector(length = n, dtype = dtype)
                        f = wp.array(data = val, dtype=vector_type)
                else:
                    f = wp.array(data = val) # implicit use the dtype defined in numpy array

                setattr(faces,key,f)
            else:
                setattr(faces,key,val)
        return faces
    

if __name__ == '__main__':
    pass