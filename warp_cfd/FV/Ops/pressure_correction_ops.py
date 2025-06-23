import warp as wp
import numpy as np
from .ops_class import Ops
from typing import Any
class Pressure_correction_Ops(Ops):
    def __init__(self,cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        super().__init__(cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype,int_dtype)

    def init(self):
        @wp.func
        def get_vector_from_array(arr:wp.array(ndim= 2,dtype=self.float_dtype),i:wp.int32):
            vec = wp.vector(arr[i][0],arr[i][1],arr[i][2],dtype= wp.float32)
            return vec



        @wp.kernel
        def _calculate_D_at_cells_kernel(D_cell:wp.array(dtype=self.vector_type),
                                         inv_A:wp.array2d(dtype=self.float_dtype),
                                         cell_structs:wp.array(dtype=self.cell_struct)
                                         ):
            i = wp.tid()

            D_vec = get_vector_from_array(inv_A,i)*cell_structs[i].volume
            D_cell[i] = D_vec




        @wp.kernel
        def _interpolate_D_to_face_kernel(D_face:wp.array(dtype=self.float_dtype),
                                          D_cell:wp.array(dtype=self.vector_type),
                                          cell_structs: wp.array(dtype= self.cell_struct),
                                          face_structs:wp.array(dtype = self.face_struct)):
            face_id = wp.tid() # K
            # D is a C,3 array This is always 3
            owner_cell_id = face_structs[face_id].adjacent_cells[0]
            face_idx = face_structs[face_id].cell_face_index[0]
            normal = cell_structs[owner_cell_id].face_normal[face_idx]
            owner_D = D_cell[owner_cell_id]
            if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
                D_face[face_id] = wp.dot(normal,wp.cw_mul(normal,owner_D))
            else:
                neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
                neighbor_D = D_cell[neighbor_cell_id]
                
                D_f = face_structs[face_id].norm_distance[0]*owner_D + face_structs[face_id].norm_distance[1]*neighbor_D
                D_face[face_id] = wp.dot(D_f,wp.cw_mul(normal,normal))
        

        @wp.kernel
        def _interpolate_p_correction_to_faces_kernel(p_correction:wp.array(dtype= self.float_dtype),
                                               p_correction_face:wp.array(dtype=self.float_dtype),
                                               face_structs:wp.array(dtype=self.face_struct)):
            face_id = wp.tid()
            adjacent_cell_ids = face_structs[face_id].adjacent_cells
            owner_id = adjacent_cell_ids[0]

            if face_structs[face_id].is_boundary:
                p_correction_face[face_id] = p_correction[owner_id]                
            else:
                # Central Difference
                neighbor_id = adjacent_cell_ids[1]
                p_correction_face[face_id] = face_structs[face_id].norm_distance[0]*p_correction[owner_id] +face_structs[face_id].norm_distance[1]*p_correction[neighbor_id]


        @wp.kernel
        def _calculate_p_correction_gradient_kernel(p_correction_face:wp.array(dtype=self.float_dtype),
                                                    p_correction_gradient:wp.array(dtype=self.float_dtype),
                                                    D_face:wp.array(dtype=self.float_dtype),
                                                    cell_structs:wp.array(dtype=self.cell_struct),
                                                    face_structs:wp.array(dtype=self.face_struct)):
            i,face_idx = wp.tid() # C,F
            row = 3*i

            face_id = cell_structs[i].faces[face_idx]
            volume = cell_structs[i].volume
            normal = cell_structs[i].face_normal[face_idx]
            area = face_structs[face_id].area
            flux = D_face[face_id]*p_correction_face[face_id]*normal*area/volume
            
            for dim in range(3):
                wp.atomic_add(p_correction_gradient,row + dim,flux[dim])

        
        @wp.kernel
        def _p_correction_laplacian_kernel(cell_structs:wp.array(dtype = self.cell_struct),
                                face_structs:wp.array(dtype = self.face_struct),
                                weights:wp.array3d(dtype=self.weight_struct),
                                D_face:wp.array(dtype=self.float_dtype)):
            i,j,output = wp.tid() # C,F,O
            face_id = cell_structs[i].faces[j]
            nu = D_face[face_id] 

            # we calculate grad(u)_f dot n_f with central differencing
            if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
                weights[i,j,output].owner = 0.
                weights[i,j,output].neighbor = 0.
            else:
                distance = wp.length(face_structs[face_id].cell_distance)
                weight = nu*(face_structs[face_id].area)/distance
                weights[i,j,output].owner = -weight
                weights[i,j,output].neighbor = weight

        self._p_correction_laplacian= _p_correction_laplacian_kernel
        self._calculate_D_at_cells = _calculate_D_at_cells_kernel #weight/ ops
        self._interpolate_D_to_face = _interpolate_D_to_face_kernel
        self._interpolate_p_correction_to_faces = _interpolate_p_correction_to_faces_kernel #
        self._calculate_p_correction_gradient = _calculate_p_correction_gradient_kernel
    

    def calculate_p_correction_weights(self,D_face,cells,faces,weights):
        wp.launch(kernel= self._p_correction_laplacian,dim = [cells.shape[0],self.faces_per_cell,1],inputs=[cells,faces,weights,D_face])
        

    def calculate_D_at_cells(self,D_cell:wp.array,inv_A:wp.array,cells:wp.array):
        assert len(inv_A.shape) == 1
        wp.launch(self._calculate_D_at_cells,dim = cells.shape[0],inputs= [D_cell,inv_A.reshape((-1,3)),cells])

    def interpolate_D_to_face(self,D_face,D_cell,cells,faces):
        wp.launch(self._interpolate_D_to_face,dim = faces.shape[0],inputs= [D_face,D_cell,cells,faces])

    def calculate_D_viscosity(self,D_cell,D_face,inv_A,cells,faces):
        self.calculate_D_at_cells(D_cell,inv_A,cells)
        self.interpolate_D_to_face(D_face,D_cell,cells,faces)

    def interpolate_p_correction_to_faces(self,p_correction,p_correction_face,faces):
        wp.launch(kernel=self._interpolate_p_correction_to_faces,dim = (faces.shape[0]),inputs = [p_correction,p_correction_face,faces])
    
    def calculate_p_correction_gradient(self,p_correction_face:wp.array,p_correction_gradient:wp.array,D_face:wp.array, cells:wp.array , faces:wp.array):
        p_correction_gradient.zero_()
        wp.launch(kernel=self._calculate_p_correction_gradient,dim = (cells.shape[0],self.faces_per_cell),inputs = [p_correction_face,p_correction_gradient,D_face,cells,faces])
    
