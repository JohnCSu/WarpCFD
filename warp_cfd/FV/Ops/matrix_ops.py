import warp as wp
from typing import Any
from warp_cfd.FV.utils import COO_Arrays
from warp.optim import linear
import warp.sparse as sparse
from warp_cfd.FV.Ops.array_ops import sub_1D_array
from .ops_class import Ops
class Matrix_Ops(Ops):
    def __init__(self,cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype = wp.float32,int_dtype = wp.int32):
        super().__init__(cell_struct,face_struct,node_struct,weight_struct,cell_properties,face_properties,num_outputs,float_dtype,int_dtype)

        self.linear_solver = linear.bicgstab
    
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
                                weights:wp.array3d(dtype= self.weight_struct),
                                output_indices:wp.array(dtype= self.int_dtype),
                                flip_sign:wp.bool):
            i = wp.tid()

            row = bsr_rows[i]
            col = bsr_columns[i]
            num_outputs = output_indices.shape[0]
            output_idx = wp.mod(row,num_outputs)
            output = output_indices[output_idx]
            cell_id = row//num_outputs
            neighbor_id = col//num_outputs

            owner_cell = cell_structs[cell_id]

            if flip_sign: # if term 
                owner_scale = -1.
                neighbor_scale = -1.
            else: # Convective so interpolation weighting or if laplacian remain
                owner_scale = 1.
                neighbor_scale = 1.

            if row == col:
                for face_idx in range(owner_cell.faces.length):
                    # if owner_cell.neighbors[face_idx] != -1: # So not boundary
                    wp.atomic_add(values,i, owner_scale*weights[cell_id,face_idx,output].owner)
            else: # Off Diagonal
                face_idx = get_face_idx(owner_cell.neighbors,neighbor_id)
                if face_idx != -1:    
                    wp.atomic_add(values,i, neighbor_scale*weights[cell_id,face_idx,output].neighbor)

        @wp.kernel
        def _calculate_RHS_values_kernel(b:wp.array(dtype=self.float_dtype),
                                        boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
                                        face_structs:wp.array(dtype=self.face_struct),
                                        weights:wp.array3d(dtype= self.weight_struct),
                                        output_indices:wp.array(dtype= self.int_dtype),
                                        flip_sign:wp.bool):
            
            i,output_idx = wp.tid() #Loop through Boundary faces

            face_id = boundary_face_ids[i]
            cell_id = face_structs[face_id].adjacent_cells[0]
            face_idx = face_structs[face_id].cell_face_index[0]
            output = output_indices[output_idx]
            row = cell_id*output_indices.shape[0] +output_idx

            if flip_sign: # Convective so moving to RHS needs -ve
                scale = -1.
            else:  # If solving for laplacian explicit terms
                scale = 1.
            # wp.atomic_add(b,row,weights[cell_id,face_idx,output].neighbor -  weights[cell_id,face_idx,output].neighbor )
            wp.atomic_add(b,row,scale*weights[cell_id,face_idx,output].neighbor )

        @wp.kernel
        def form_p_grad_vector_kernel(p_grad:wp.array(dtype=self.float_dtype),cell_gradients:wp.array2d(dtype=self.vector_type),
                                            cell_structs: wp.array(dtype=self.cell_struct),density:self.float_dtype ):
                i,dim = wp.tid() # Go by 
                D = wp.static(self.dimension)
                p = wp.static(self.int_dtype(3))
                row = i*D + dim
                # wp.printf('n1 %f n2 %f n3 %f \n',cell_structs[i].grads[p][0],cell_structs[i].grads[p][1],cell_structs[i].grads[p][2])
                p_grad[row] =  (cell_gradients[i,p][dim])*cell_structs[i].volume
            


        @wp.kernel
        def _update_p_correction_rows_kernel(bsr_row_offsets:wp.array(dtype= self.int_dtype),
                                bsr_columns: wp.array(dtype= self.int_dtype),
                                values:wp.array(dtype=self.float_dtype),
                                b:wp.array(dtype=self.float_dtype),
                                fixed_cells_id:wp.array(dtype=self.int_dtype)):
            
            i = wp.tid() # We go through all Cells with fixed pressure values
            
            #For now only for pressure so cell_id = row
            cell_id = fixed_cells_id[i]
            col_start,col_end = bsr_row_offsets[cell_id], bsr_row_offsets[cell_id+1]
            for nnz_idx in range(col_start,col_end):
                column = bsr_columns[nnz_idx]
                if cell_id == column: #Diagonal
                    values[nnz_idx] = 1.
                else: # Off diagonal
                    values[nnz_idx] = 0.
            
            b[cell_id] = 0. # Replace the div u with 0 to indicate no pressure correction

        

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



        self._form_p_grad_vector_kernel= form_p_grad_vector_kernel
        '''attribute to call method with which to form the pressure gradient vector'''
        self._fill_matrix_vector_from_outputs = _fill_matrix_vector_from_outputs
        self._fill_outputs_from_matrix_vector = _fill_outputs_from_matrix_vector
        # self._calculate_BSR_matrix_and_RHS = calculate_BSR_matrix_and_RHS_kernel
        self._calculate_BSR_values = _calculate_BSR_values_kernel
        self._calculate_RHS_values = _calculate_RHS_values_kernel

        self._get_matrix_indices = get_matrix_indices

        

        self._update_p_correction_rows = _update_p_correction_rows_kernel #matrix ops
    
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
    

    def calculate_BSR_matrix_and_RHS(self,BSR_matrix:sparse.BsrMatrix,cells,faces,weights,output_indices:wp.array,flip_sign:bool,b:wp.array = None,rows =None):
        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        assert rows.shape[0] == BSR_matrix.values.shape[0]
        
        wp.launch(kernel=self._calculate_BSR_values,dim = BSR_matrix.values.shape[0],inputs=[rows,
                                                                                             BSR_matrix.columns,
                                                                                             BSR_matrix.values,
                                                                                             cells,
                                                                                             weights,
                                                                                             output_indices,
                                                                                             flip_sign])
        
        if b is not None:
            boundary_ids = self.face_properties.boundary_face_ids
            wp.launch(kernel= self._calculate_RHS_values, dim = (boundary_ids.shape[0],output_indices.shape[0]),inputs = [b,
                                                                                                                    boundary_ids,
                                                                                                                    faces,
                                                                                                                    weights,
                                                                                                                    output_indices,
                                                                                                                    not flip_sign,

            ])
    def form_p_grad_vector(self,grad_P,cell_gradients,cells,density):
        wp.launch(kernel=self._form_p_grad_vector_kernel, dim = [cells.shape[0],self.dimension], inputs = [grad_P,cell_gradients,cells,density])
        
    
    def get_b_vector(self,H,grad_P,b):
        sub_1D_array(H,grad_P,b)
    
    def solve_Axb(self,A:sparse.BsrMatrix,x:wp.array,b:wp.array):
        M = linear.preconditioner(A)
        return self.linear_solver(A =A,M= M,tol = 1.e-6, b = b,x = x,maxiter=500)
    
    def update_p_correction_rows(self,bsr_matrix:sparse.BsrMatrix,div_u:wp.array):
        fixed_cells = self.cell_properties.fixed_cells
        wp.launch(kernel=self._update_p_correction_rows,dim = fixed_cells.shape[0], inputs=[bsr_matrix.offsets,bsr_matrix.columns,bsr_matrix.values,div_u,fixed_cells])
