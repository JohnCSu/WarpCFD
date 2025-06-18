import warp as wp
from warp_cfd.FV.Ops.mesh_ops import Mesh_Ops 
from warp_cfd.FV.Ops.weight_ops import Weights_Ops
from warp_cfd.FV.Ops.matrix_ops import Matrix_Ops
from warp import sparse
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.Ops.array_ops import inv_1D_array
class intermediate_velocity_step():
    def __init__(self,mesh_ops:Mesh_Ops,weight_ops:Weights_Ops,matrix_ops:Matrix_Ops) -> None:
        self.mesh_ops = mesh_ops
        self.weight_ops = weight_ops
        self.matrix_ops = matrix_ops
        
        self.cell_properties = mesh_ops.cell_properties
        self.face_properties = mesh_ops.face_properties
        self.num_cells = mesh_ops.num_cells
        self.faces_per_cell = mesh_ops.faces_per_cell

        self.float_dtype = mesh_ops.float_dtype
        self.int_dtype = mesh_ops.int_dtype
        self.dimension = mesh_ops.dimension

    def init(self,cells,faces):

       
        self.vel_COO_array = COO_Arrays(self.cell_properties.nnz_per_cell,self.dimension,self.float_dtype,self.int_dtype)
        '''Arrays for Sparse Matrix for U step'''
        # WE mak A*u = B matrix A. In 3D we have 3
        self.sparse_dimensions = (self.num_cells*self.dimension,self.num_cells*self.dimension)
        self.H = wp.zeros(shape=(self.dimension*self.num_cells),dtype=self.float_dtype)
        self.grad_P = wp.zeros_like(src = self.H)
        self.b = wp.zeros_like(src = self.H)
        self.intermediate_vel = wp.zeros_like(src = self.H)
        self.initial_vel = wp.zeros_like(src = self.H)

        self.matrix_ops.calculate_BSR_matrix_indices(self.vel_COO_array,cells,faces,num_outputs = 3)
        '''
        Each Cell results in 3 rows. Each Row would have 1 + num neighbors. We can actually calculate this beforehand
        That means each cell gives 3*(1+num_neighbors) of nnz. So total nnz = 3*(1+num_neighbors)*num_cells 
        '''
        
        self.vel_matrix = sparse.bsr_from_triplets(*self.sparse_dimensions,
                                                      rows = self.vel_COO_array.rows,
                                                      columns = self.vel_COO_array.cols,
                                                      values = self.vel_COO_array.values,
                                                      prune_numerical_zeros= False)
        
        self.inv_A = wp.zeros(shape = self.vel_matrix.shape[0],dtype = self.float_dtype)
        self.vel_matrix_rows = self.vel_matrix.uncompress_rows()
    

    def solve(self,cells,faces,weights,density,vel_indices):
        self.matrix_ops.calculate_BSR_matrix_and_RHS(self.vel_matrix,cells,faces,weights,output_indices=vel_indices,b = self.H,rows= self.vel_matrix_rows)
        self.matrix_ops.fill_x_array_from_struct(self.intermediate_vel,cells,vel_indices)
        self.matrix_ops.form_p_grad_vector(self.grad_P,cells,density)
        self.matrix_ops.get_b_vector(self.H,self.grad_P,self.b)
        
        wp.copy(self.intermediate_vel,self.initial_vel) # Use velocity at current iteration as initial guess
        result = self.matrix_ops.solve_Axb(A = self.vel_matrix,x = self.intermediate_vel,b = self.b)
        print(result)
        self.matrix_ops.fill_struct_from_x_array(self.intermediate_vel,cells,vel_indices)
        self.inv_A = sparse.bsr_get_diag(self.vel_matrix)
        inv_1D_array(self.inv_A,self.inv_A) # Invert A i.e. 1/a_i

        return self.vel_matrix,self.initial_vel,self.b,self.intermediate_vel,self.inv_A