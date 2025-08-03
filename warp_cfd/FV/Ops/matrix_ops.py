import warp as wp
from typing import Any
from warp_cfd.FV.utils import COO_Arrays
from warp.optim import linear
import warp.sparse as sparse
from warp_cfd.FV.Ops.array_ops import sub_1D_array
from .ops_class import Ops
class Matrix_Ops(Ops):
    def __init__(self,cell_struct,face_struct,node_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        super().__init__(cell_struct,face_struct,node_struct,cell_properties,face_properties,num_outputs,float_dtype,int_dtype)
    
    def init(self):
        
        @wp.kernel
        def get_matrix_indices( rows:wp.array(dtype=self.int_dtype),
                                cols:wp.array(dtype=self.int_dtype),
                                offsets:wp.array(dtype=self.int_dtype),
                                cell_structs: wp.array(dtype=self.cell_struct),
                                face_structs:wp.array(dtype=self.face_struct),
                                num_outputs: self.int_dtype):
            i,nnz_cell_offset,output = wp.tid() # Loop C,NNZ,D (set to 3)
            D = num_outputs
            cell_offset = offsets[i]
            num_nnz_per_cell = cell_structs[i].nnz
            # wp.printf('i:%d offset:%d output: %d \n',i,nnz_cell_offset,output)   
            if nnz_cell_offset == 0: # We are weighing the diagonal
                total_offset = cell_offset + (num_nnz_per_cell)*output
                rows[total_offset] = D*i + output
                #Because of structure, if 0 then we just record that the diagonal is needed. Updates to values is made when we iterate over face_idx
                cols[total_offset] = rows[total_offset]
                # wp.printf('i:%d offset:%d num_nnz_per_cell %d nnz_cell_offset:%d output:%d r:%d c:%d t:%d \n', i,cell_offset,num_nnz_per_cell,nnz_cell_offset,output,rows[total_offset],cols[total_offset],total_offset)  
            else:
                # Find the neighbor cell id corresponding to the face idx which gives us the column
                face_idx = nnz_cell_offset - 1 # 0 is considered the diagonal (self)
                face_id = cell_structs[i].faces[face_idx]
                
                if face_structs[face_id].is_boundary == 0: # Internal Face
                    face_offset = cell_structs[i].face_offset_index[face_idx]
                    total_offset = cell_offset + (num_nnz_per_cell)*output + face_offset # Need Face offset so for each face index from 1->(F+1)
                    rows[total_offset] = D*i + output
                    neighbor_cell_id = cell_structs[i].neighbors[face_idx]
                    cols[total_offset] = D*neighbor_cell_id + output
                    # wp.printf('i:%d offset:%d num_nnz_per_cell %d nnz_cell_offset:%d output:%d r:%d c:%d t:%d \n', i,cell_offset,num_nnz_per_cell,nnz_cell_offset,output,rows[total_offset],cols[total_offset],total_offset)  
            # wp.printf('i:%d offset:%d num_nnz_per_cell %d output:%d \n', i,cell_offset,nnz_cell_offset,output)
        

        @wp.func
        def get_face_idx(vec1:Any,id:int):
            for i in range(vec1.length):
                if vec1[i] == id:
                    return i
            
            return -1
        
        @wp.kernel
        def _calculate_BSR_values_kernel(bsr_rows:wp.array(dtype=int),
                                bsr_columns:wp.array(dtype=int),
                                values:wp.array(dtype= self.float_dtype),
                                cell_structs:wp.array(dtype= self.cell_struct),
                                weights:wp.array4d(dtype= self.float_dtype),
                                output_indices:wp.array(dtype= self.int_dtype),
                                scale:self.float_dtype):
            i = wp.tid()

            row = bsr_rows[i]
            col = bsr_columns[i]
            num_outputs = output_indices.shape[0]
            output_idx = wp.mod(row,num_outputs)
            output = output_indices[output_idx]
            cell_id = row//num_outputs
            neighbor_id = col//num_outputs

            owner_cell = cell_structs[cell_id]

            if row == col:
                for face_idx in range(owner_cell.faces.length):
                    # if owner_cell.neighbors[face_idx] != -1: # So not boundary
                    wp.atomic_add(values,i, scale*weights[cell_id,face_idx,output,0])
            else: # Off Diagonal
                face_idx = get_face_idx(owner_cell.neighbors,neighbor_id)
                if face_idx != -1:    
                    wp.atomic_add(values,i, scale*weights[cell_id,face_idx,output,1])

        @wp.kernel
        def _calculate_RHS_values_kernel(b:wp.array(dtype=self.float_dtype),
                                        boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
                                        face_structs:wp.array(dtype=self.face_struct),
                                        weights:wp.array4d(dtype= self.float_dtype),
                                        output_indices:wp.array(dtype= self.int_dtype),
                                        scale:self.float_dtype):
            
            i,output_idx = wp.tid() #Loop through Boundary faces

            face_id = boundary_face_ids[i]
            cell_id = face_structs[face_id].adjacent_cells[0]
            face_idx = face_structs[face_id].cell_face_index[0]
            output = output_indices[output_idx]
            row = cell_id*output_indices.shape[0] +output_idx

            # wp.atomic_add(b,row,weights[cell_id,face_idx,output].neighbor -  weights[cell_id,face_idx,output].neighbor )
            wp.atomic_add(b,row,-scale*weights[cell_id,face_idx,output,2] )

        @wp.kernel
        def form_p_grad_vector_kernel(p_grad:wp.array(dtype=self.float_dtype),cell_gradients:wp.array2d(dtype=self.vector_type),
                                            cell_structs: wp.array(dtype=self.cell_struct),density:self.float_dtype ):
                i,dim = wp.tid() # Go by 
                D = wp.static(self.dimension)
                p = 3
                row = i*D + dim
                # wp.printf('n1 %f n2 %f n3 %f \n',cell_structs[i].grads[p][0],cell_structs[i].grads[p][1],cell_structs[i].grads[p][2])
                p_grad[row] =  (cell_gradients[i,p][dim])*cell_structs[i].volume
            


        @wp.kernel
        def replace_row_kernel(bsr_row_offsets:wp.array(dtype= self.int_dtype),
                                bsr_columns: wp.array(dtype= self.int_dtype),
                                bsr_values:wp.array(dtype=self.float_dtype),
                                b:wp.array(dtype=self.float_dtype),
                                row_ids:wp.array(dtype=self.int_dtype),
                                rhs_values:wp.array(dtype=self.float_dtype)):
            
            i = wp.tid() # We go through all Cells with fixed pressure values
            
            #For now only for pressure so cell_id = row
            row_id = row_ids[i]
            col_start,col_end = bsr_row_offsets[row_id], bsr_row_offsets[row_id+1]
            for nnz_idx in range(col_start,col_end):
                column = bsr_columns[nnz_idx]
                if row_id == column: #Diagonal
                    bsr_values[nnz_idx] = self.float_dtype(1.)
                else: # Off diagonal
                    bsr_values[nnz_idx] = self.float_dtype(0.)
            
            b[row_id] = rhs_values[row_id] # Replace the row with the specified value

        

        @wp.kernel
        def _fill_matrix_vector_from_outputs(arr:wp.array(dtype=self.float_dtype),cell_values:wp.array2d(dtype=self.float_dtype),output_indices:wp.array(dtype = self.int_dtype)):
            i,output_idx = wp.tid()
            num_outputs = output_indices.shape[0]
            output = output_indices[output_idx]
            row = num_outputs*i + output_idx
            arr[row] = cell_values[i,output]


        @wp.kernel
        def _fill_outputs_from_matrix_vector(arr:wp.array(dtype=self.float_dtype),cell_values:wp.array2d(dtype=self.float_dtype),output_indices:wp.array(dtype = self.int_dtype)):
            i,output_idx = wp.tid()
            num_outputs = output_indices.shape[0]
            output = output_indices[output_idx]
            row = num_outputs*i + output_idx
            cell_values[i,output] = arr[row]



        @wp.kernel
        def _implicit_relaxation(bsr_rows:wp.array(dtype=int),
                                bsr_columns:wp.array(dtype=int),
                                values:wp.array(dtype= self.float_dtype),
                                b:wp.array(dtype=self.float_dtype),
                                cell_values:wp.array2d(dtype=self.float_dtype),
                                alpha:self.float_dtype,
                                output_indices:wp.array(dtype=int)):
            ''' Can be optimised to only take diag ahead of time'''
            i = wp.tid()

            row = bsr_rows[i]
            col = bsr_columns[i]

            if row == col: # Diag
                # Row//num_outputs == Cell_Id ?
                num_outputs = output_indices.shape[0]
                output_idx = wp.mod(row,num_outputs)
                
                output = output_indices[output_idx]
                cell_id = row//num_outputs
                cell_value = cell_values[cell_id,output]
                
                b[row] +=  (self.float_dtype(1.)-alpha)/alpha*values[i]*cell_value
                values[i] = values[i]/alpha
                
            # For Matrix:  ap/alpha and LHS: add (1-alpha)/alpha*ap*u_old

        @wp.kernel
        def add_RHS_vector(rhs:wp.array(dtype=self.float_dtype),
                           arr:wp.array(dtype=self.float_dtype),
                           scale:self.float_dtype):
            
            i = wp.tid()
            wp.atomic_add(rhs,i,scale*arr[i])
        
        self._form_p_grad_vector_kernel= form_p_grad_vector_kernel
        '''attribute to call method with which to form the pressure gradient vector'''
        self._fill_matrix_vector_from_outputs = _fill_matrix_vector_from_outputs
        self._fill_outputs_from_matrix_vector = _fill_outputs_from_matrix_vector
        # self._calculate_BSR_matrix_and_RHS = calculate_BSR_matrix_and_RHS_kernel
        self._calculate_BSR_values = _calculate_BSR_values_kernel
        self._calculate_RHS_values = _calculate_RHS_values_kernel
        self._add_to_RHS = add_RHS_vector
        self._get_matrix_indices = get_matrix_indices

        self._implicit_relaxation = _implicit_relaxation

        self._replace_row = replace_row_kernel #matrix ops


    def implicit_relaxation(self,BSR_matrix:sparse.BsrMatrix,b,cell_values,alpha,output_indices,rows = None):
        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        cols,values = BSR_matrix.columns,BSR_matrix.values
        wp.launch(kernel=self._implicit_relaxation,dim = rows.shape[0],inputs = [rows,cols,values,b,cell_values,alpha,output_indices])
    
    def fill_matrix_vector_from_outputs(self,cell_values,matrix_arr,output_indices):
        wp.launch(kernel=self._fill_matrix_vector_from_outputs,dim = (self.num_cells,output_indices.shape[0]),inputs = [matrix_arr,cell_values,output_indices] )

    def fill_outputs_from_matrix_vector(self,cell_values,matrix_arr,output_indices):
        wp.launch(kernel=self._fill_outputs_from_matrix_vector,dim = (self.num_cells,output_indices.shape[0]),inputs = [matrix_arr,cell_values,output_indices] )
    
    def calculate_BSR_matrix_indices(self,COO_array:COO_Arrays,cells,faces,num_outputs:int):
        wp.launch(kernel=self._get_matrix_indices,dim = [cells.shape[0],self.faces_per_cell+1,num_outputs],inputs = [COO_array.rows,
                                                                                                                  COO_array.cols,
                                                                                                                  COO_array.offsets,
                                                                                                                  cells,
                                                                                                                  faces,
                                                                                                                  num_outputs
                                                                                                                  ])
    def calculate_Implicit_LHS(self,BSR_matrix:sparse.BsrMatrix,cells,weights,scale,rows =None):

        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        assert rows.shape[0] == BSR_matrix.values.shape[0]

        output_indices = wp.array([i for i in range(weights.shape[-2])],dtype= int)
        wp.launch(kernel=self._calculate_BSR_values,dim = BSR_matrix.values.shape[0],inputs=[rows,
                                                                                             BSR_matrix.columns,
                                                                                             BSR_matrix.values,
                                                                                             cells,
                                                                                             weights,
                                                                                             output_indices,
                                                                                             scale])
    def calculate_Implicit_RHS(self,b,faces,weights,scale:float):
        
        output_indices = wp.array([i for i in range(weights.shape[-2])],dtype= int)
        boundary_ids = self.face_properties.boundary_face_ids

        wp.launch(kernel= self._calculate_RHS_values, dim = (boundary_ids.shape[0],output_indices.shape[0]),inputs = [b,
                                                                                                                boundary_ids,
                                                                                                                faces,
                                                                                                                weights,
                                                                                                                output_indices,
                                                                                                                scale,
        ])

    def add_to_RHS(self,rhs:wp.array,arr:wp.array,scale:float):
        assert rhs.shape == arr.shape
        wp.launch(self._add_to_RHS,dim = rhs.shape,inputs=[rhs,arr,scale])

    def form_p_grad_vector(self,grad_P,cell_gradients,cells,density):
        wp.launch(kernel=self._form_p_grad_vector_kernel, dim = [cells.shape[0],self.dimension], inputs = [grad_P,cell_gradients,cells,density])
        
    
    def get_b_vector(self,H,grad_P,b):
        sub_1D_array(H,grad_P,b)
    
    def solve_Axb(self,A:sparse.BsrMatrix,x:wp.array,b:wp.array,linear_solver,tol = 1.e-7,max_iter = 500):
        M = linear.preconditioner(A)
        results =linear_solver(A =A,M= M,atol = tol, b = b,x = x,maxiter=max_iter) 
        
        return results
    
    def replace_row(self,bsr_matrix:sparse.BsrMatrix,rhs:wp.array,row_id:wp.array,value:wp.array):
        wp.launch(kernel=self._replace_row,dim = row_id.shape, inputs=[bsr_matrix.offsets,bsr_matrix.columns,bsr_matrix.values,rhs,row_id,value])




