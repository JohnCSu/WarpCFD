import numpy as np
from warp_cfd.FV.field import Field
import warp as wp

class Boundary():
    boundary_ids: np.ndarray[int]
    boundary_type: np.ndarray[int]
    boundary_values:np.ndarray[float]
    groups: dict[str,np.ndarray]
    num_outputs: int

    def __init__(self,fields:dict[str,Field],groups,boundary_ids,face_areas=None,face_normals=None,face_centroids=None,float_type = wp.float32) -> None:

        self.fields = fields
        self.num_outputs = len(list(self.fields.keys()))
        self.groups = groups
        self.boundary_ids = boundary_ids
        self.boundary_values = np.zeros(shape = (len(boundary_ids),self.num_outputs),dtype=wp.dtype_to_numpy(float_type))
        self.boundary_type = np.zeros(shape = self.boundary_values.shape, dtype = np.uint8)
        self.float_dtype = float_type

        self._boundary_types = {
            'dirichlet': 1,
            'vonNeumann': 2,
        }

        # self.face_areas = face_areas[boundary_ids]
        # self.face_normals = face_normals[boundary_ids]
        # self.face_centroids = face_centroids[boundary_ids]

    def set_value(self,face_ids: np.ndarray,boundary_type = 'dirchlet',**output_vars):
        
        face_ids = self.get_face_id_array(face_ids)
        
        assert np.all(np.isin(face_ids,self.boundary_ids)), 'One or more of the provided face id is not an external face valid for applying boundary conditions'
        ''' we either have a float for all in face_ids or an array of same len'''

        assert output_vars, 'You must specify some sort of output variable'

        face_ids = np.where(np.isin(self.boundary_ids, face_ids))[0]

        for output_name,value in output_vars.items():
            output = self.fields[output_name]
            self.boundary_values[face_ids,output.index] = value
            self.boundary_type[face_ids,output.index] = self._boundary_types[boundary_type] 
                    
    def get_face_id_array(self,face_ids:str |int | list |tuple |np.ndarray):
        if isinstance(face_ids,str):
            assert face_ids in list(self.groups.keys()), 'Specified group name does not exist'
            return self.groups[face_ids].ids

        elif isinstance(face_ids,(int,list,tuple,np.ndarray)):
            if not isinstance(face_ids,(np.ndarray)):
                return np.array([face_ids],np.int32)
            else: # Is already Numpy array
                return face_ids

        else:
            raise ValueError(f'face_ids can be type string,int,list,tuple, or np.ndarray got {type(face_ids)} instead')

    def gradient_BC(self,face_ids:str | int | list |tuple |np.ndarray,**output_vars):
        '''
        Set the gradient At a Boundary Face. if the input is a vector (i.e an array/list of 3 floats) then the gradient is projected onto the face with the face's normal
        
        If a float is specified, it is assumed to is the normal gradient
        '''

        for output_var in output_vars.values():
            if isinstance(output_var,(np.ndarray,list,tuple)):
                pass
        self.set_value(face_ids,**output_vars,boundary_type='vonNeumann')


    def fixed_BC(self,face_ids:str | int | list |tuple |np.ndarray,**output_vars):
        self.set_value(face_ids,**output_vars,boundary_type='dirichlet')

    def velocity_BC(self,face_ids,u,v,w):
        self.fixed_BC(face_ids,u=u,v=v,w=w)
        self.gradient_BC(face_ids,p = 0.)

    def pressure_BC(self,face_ids,p,backflow=None):
        self.fixed_BC(face_ids,p = p)
        self.gradient_BC(face_ids,u=0.,v=0.,w=0.)

    def mass_flow_BC(self,face_ids,mass_flow,inlet = True):
        raise ValueError('not yet implemented')
    
    def no_slip_wall(self,face_ids):
        self.fixed_BC(face_ids,u=0.,v=0.,w=0.)
        self.gradient_BC(face_ids,p = 0.)

    def slip_wall(self,face_ids):
        self.gradient_BC(face_ids,u=0.,v=0.,w=0.,p = 0.)
        

    def check_and_to_warp(self):
        '''
        Final Checks before converting to warp arrays
        '''

        if np.any(self.boundary_type < 1):
            raise ValueError(f'The following faces have not been assigned a boundary condition: {self.boundary_ids[ (self.boundary_type < 1 ).all(axis= 1)]}')  

        if np.any(self.boundary_type > 3):
            raise ValueError(f'Somehow you have a boundary type with a value of 3. Only 1 and 2 arw allowed')

        self.boundary_ids = wp.array(self.boundary_ids,dtype= wp.int32)
        self.boundary_type = wp.array(self.boundary_type,dtype= wp.uint8)
        self.boundary_values = wp.array(self.boundary_values,dtype= self.float_dtype)
        