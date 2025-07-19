import warp as wp
from warp_cfd.FV.Ops import Mesh_Ops 
from warp_cfd.FV.Ops import Weights_Ops
from warp_cfd.FV.Ops import Matrix_Ops
from warp_cfd.FV.Ops import Pressure_correction_Ops
from warp import sparse
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.Ops.array_ops import inv_1D_array,mult_1D_array,mult_scalar_1D_array,add_1D_array
import numpy as np

class pressure_correction_step():
    def __init__(self,mesh_ops:Mesh_Ops,weight_ops:Weights_Ops,matrix_ops:Matrix_Ops,pressure_correction_ops:Pressure_correction_Ops) -> None:
        self.mesh_ops = mesh_ops
        self.weight_ops = weight_ops
        self.matrix_ops = matrix_ops
        
        self.cell_properties = mesh_ops.cell_properties
        self.face_properties = mesh_ops.face_properties
        self.num_cells = mesh_ops.num_cells
        self.num_faces = mesh_ops.num_faces
        self.faces_per_cell = mesh_ops.faces_per_cell

        self.float_dtype = mesh_ops.float_dtype
        self.int_dtype = mesh_ops.int_dtype
        self.dimension = mesh_ops.dimension
        self.pressure_correction_ops = pressure_correction_ops

    def init(self,cells,faces):
        self.p = wp.zeros(shape= self.num_cells,dtype=self.float_dtype)
        self.p_relaxation_factor = 0.3
        
        self.p_correction_weight_struct = self.mesh_ops.weight_struct

        self.weights = wp.array(shape = (self.num_cells,self.faces_per_cell,1),dtype= self.p_correction_weight_struct)

        self.p_COO_array = COO_Arrays(self.cell_properties.nnz_per_cell,1,self.float_dtype,self.int_dtype)
        self.div_u = wp.zeros(shape = self.num_cells,dtype = self.float_dtype)
        self.p_correction = wp.zeros(shape=self.num_cells, dtype= self.float_dtype)
        self.p_correction_face = wp.zeros(shape= self.num_faces, dtype= self.float_dtype)
        self.velocity_correction = wp.zeros(shape = self.num_cells*self.dimension, dtype= self.float_dtype)
        '''Arrays for Sparse Matrix fro pressure correction'''
        self.matrix_ops.calculate_BSR_matrix_indices(self.p_COO_array,cells,faces,1)
        self.p_correction_matrix = sparse.bsr_from_triplets(self.num_cells,self.num_cells,
                                                            rows = self.p_COO_array.rows,
                                                            columns = self.p_COO_array.cols,
                                                            values = self.p_COO_array.values,
                                                            prune_numerical_zeros= False)
        self.p_correction_matrix_rows = self.p_correction_matrix.uncompress_rows()
        self.vel_indices = wp.array([0,1,2],dtype=wp.int32)
        self.p_index = wp.array([3],dtype= wp.int32)
        self.p_correction_index = wp.zeros(1,dtype=self.int_dtype)
    
    

        
    def calculate_pressure_gradient(self,p_correction,p_correction_face:wp.array,p_correction_gradient:wp.array,D_face:wp.array, cells:wp.array , faces:wp.array):
        '''
        Calculate u_correction = 1/a_p*grad_p_correction Using Green Gauss and places it in to p_correction_gradient to use as velocity correction
        '''
        self.pressure_correction_ops.interpolate_p_correction_to_faces(p_correction,p_correction_face,faces)
        self.pressure_correction_ops.calculate_p_correction_gradient(p_correction_face,p_correction_gradient,D_face,cells,faces)


    def reset(self):
        self.p_correction_matrix.values.zero_()
        self.p_correction.zero_()
        self.div_u.zero_()


    def solve(self,weights,intermediate_vel,corrected_vel,cell_values,D_face,mass_fluxes,cells,faces):
        self.reset()
        vel_indices = self.vel_indices
        p_index = self.p_index
        
        self.mesh_ops.calculate_divergence(mass_fluxes,cells,self.div_u)
        # print(self.div_u.numpy())
        # D_face2 = wp.ones_like(D_face)
        if weights is None:
            weights = self.weights
            self.pressure_correction_ops.calculate_p_correction_weights(D_face,cells,faces,weights)
        
       
        self.matrix_ops.calculate_BSR_matrix(self.p_correction_matrix,cells,weights,output_indices=self.p_correction_index,rows = self.p_correction_matrix_rows,flip_sign= False) # only calculate BSR Values
        self.matrix_ops.update_p_correction_rows(self.p_correction_matrix,self.div_u)
        # print(self.p_correction_matrix.values.numpy().mean())
        results = self.matrix_ops.solve_Axb(self.p_correction_matrix,self.p_correction,self.div_u)
        # print(results)
        # print(self.p_correction)

        # Calculate Vp/ap grap P
        self.calculate_pressure_gradient(self.p_correction,self.p_correction_face,self.velocity_correction,D_face,cells,faces)
        # print('hi',self.velocity_correction)
        # Calc velocity correction :=  - Vp/ap *gradp'
        mult_scalar_1D_array(self.velocity_correction,-1.,self.velocity_correction)

        # add correction to intermediate velocity
        add_1D_array(intermediate_vel,self.velocity_correction,corrected_vel)

        self.matrix_ops.fill_outputs_from_matrix_vector(cell_values,corrected_vel,vel_indices)

        # Update pressure p_old +relax*p'
        self.matrix_ops.fill_matrix_vector_from_outputs(cell_values,self.p,p_index)
        mult_scalar_1D_array(self.p_correction,self.p_relaxation_factor,self.p_correction)
        add_1D_array(self.p,self.p_correction,self.p)

        self.matrix_ops.fill_outputs_from_matrix_vector(cell_values,self.p,p_index)

        return self.p,self.div_u,self.p_correction,self.velocity_correction