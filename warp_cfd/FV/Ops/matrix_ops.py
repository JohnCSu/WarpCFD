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
                                output_indices:wp.array(dtype= self.int_dtype)):
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
                    wp.atomic_add(values,i, -weights[cell_id,face_idx,output].laplacian_owner + weights[cell_id,face_idx,output].convection_owner)
            else: # Off Diagonal
                face_idx = get_face_idx(owner_cell.neighbors,neighbor_id)
                if face_idx != -1:    
                    wp.atomic_add(values,i, -weights[cell_id,face_idx,output].laplacian_neighbor + weights[cell_id,face_idx,output].convection_neighbor)

        @wp.kernel
        def _calculate_RHS_values_kernel(b:wp.array(dtype=self.float_dtype),
                                        boundary_face_ids:wp.array(dtype= self.face_properties.boundary_face_ids.dtype),
                                        face_structs:wp.array(dtype=self.face_struct),
                                        weights:wp.array3d(dtype= self.weight_struct),
                                        output_indices:wp.array(dtype= self.int_dtype)):
            
            i,output_idx = wp.tid() #Loop through Boundary faces

            face_id = boundary_face_ids[i]
            cell_id = face_structs[face_id].adjacent_cells[0]
            face_idx = face_structs[face_id].cell_face_index[0]
            output = output_indices[output_idx]
            row = cell_id*output_indices.shape[0] +output_idx

            wp.atomic_add(b,row,weights[cell_id,face_idx,output].laplacian_neighbor -  weights[cell_id,face_idx,output].convection_neighbor )


        @wp.kernel
        def form_p_grad_vector_kernel(b:wp.array(dtype=self.float_dtype),
                                            cell_structs: wp.array(dtype=self.cell_struct),density:self.float_dtype):
                i,dim = wp.tid() # Go by 
                D = wp.static(self.dimension)
                p = wp.static(self.int_dtype(3))
                row = i*D + dim
                # wp.printf('n1 %f n2 %f n3 %f \n',cell_structs[i].grads[p][0],cell_structs[i].grads[p][1],cell_structs[i].grads[p][2])
                b[row] =  cell_structs[i].gradients[p][dim]/density


        @wp.kernel
        def fill_x_array_from_struct_kernel(x:wp.array(dtype=self.float_dtype),cell_structs:wp.array(dtype=self.cell_struct),output_indices:wp.array(dtype=self.int_dtype)):
            i,output_idx = wp.tid() # C,O
            
            output = output_indices[output_idx]
            num_outputs = output_indices.shape[0]
            row = i*num_outputs + output_idx
            x[row] = cell_structs[i].values[output]

        @wp.kernel
        def fill_struct_from_x_array_kernel(x:wp.array(dtype=self.float_dtype),cell_structs:wp.array(dtype=self.cell_struct),output_indices:wp.array(dtype=self.int_dtype)):

            i,output_idx = wp.tid() # C,O
            
            output = output_indices[output_idx]
            num_outputs = output_indices.shape[0]
            row = i*num_outputs + output_idx
            cell_structs[i].values[output] = x[row] 
            
        @wp.kernel
        def _massflux_to_array_kernel(cell_structs:wp.array(dtype=self.cell_struct),out:wp.array2d(dtype = self.float_dtype)):
            i,j = wp.tid()
            out[i,j] = cell_structs[i].mass_fluxes[j]


        self._form_p_grad_vector_kernel= form_p_grad_vector_kernel
        '''attribute to call method with which to form the pressure gradient vector'''
        self._massflux_to_array = _massflux_to_array_kernel
        # self._calculate_BSR_matrix_and_RHS = calculate_BSR_matrix_and_RHS_kernel
        self._calculate_BSR_values = _calculate_BSR_values_kernel
        self._calculate_RHS_values = _calculate_RHS_values_kernel

        self._get_matrix_indices = get_matrix_indices

        self._fill_x_array_from_struct=fill_x_array_from_struct_kernel
        self._fill_struct_from_x_array = fill_struct_from_x_array_kernel

    
    def calculate_BSR_matrix_indices(self,COO_array:COO_Arrays,cells,faces,num_outputs:int):
        wp.launch(kernel=self._get_matrix_indices,dim = [cells.shape[0],self.faces_per_cell+1,num_outputs],inputs = [COO_array.rows,
                                                                                                                  COO_array.cols,
                                                                                                                  COO_array.offsets,
                                                                                                                  cells,
                                                                                                                  faces,
                                                                                                                  num_outputs
                                                                                                                  ])


    def calculate_BSR_matrix_and_RHS(self,BSR_matrix:sparse.BsrMatrix,cells,faces,weights,output_indices:wp.array,b:wp.array = None,rows =None):
        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        assert rows.shape[0] == BSR_matrix.values.shape[0]
        BSR_matrix.values.zero_() # Reset to zero
        wp.launch(kernel=self._calculate_BSR_values,dim = BSR_matrix.values.shape[0],inputs=[rows,
                                                                                             BSR_matrix.columns,
                                                                                             BSR_matrix.values,
                                                                                             cells,
                                                                                             weights,
                                                                                             output_indices])
        
        if b is not None:
            b.zero_()
            boundary_ids = self.face_properties.boundary_face_ids
            wp.launch(kernel= self._calculate_RHS_values, dim = (boundary_ids.shape[0],output_indices.shape[0]),inputs = [b,
                                                                                                                    boundary_ids,
                                                                                                                    faces,
                                                                                                                    weights,
                                                                                                                    output_indices,

            ])
    def form_p_grad_vector(self,grad_P,cells,density):
        wp.launch(kernel=self._form_p_grad_vector_kernel, dim = [cells.shape[0],self.dimension], inputs = [grad_P,cells,density])
        
    def fill_x_array_from_struct(self,x,cells,output_indices):
        
        assert x.shape[0] == cells.shape[0]*output_indices.shape[0]
        wp.launch(kernel=self._fill_x_array_from_struct,dim = [cells.shape[0],output_indices.shape[0]],inputs = [x,cells,output_indices])
    
    def fill_struct_from_x_array(self,x,cells,output_indices):
        assert x.shape[0] == cells.shape[0]*output_indices.shape[0]
        wp.launch(kernel=self._fill_struct_from_x_array,dim = [cells.shape[0],output_indices.shape[0]],inputs = [x,cells,output_indices])

    def get_b_vector(self,H,grad_P,b):
        sub_1D_array(H,grad_P,b)
    
    def solve_Axb(self,A:sparse.BsrMatrix,x:wp.array,b:wp.array):
        M = linear.preconditioner(A)
        return self.linear_solver(A =A,M= M, b = b,x = x,maxiter=500)
    
    def massflux_to_array(self,arr:wp.array,cells):
        assert len(arr.shape) == 2 and arr.shape[0] == cells.shape[0] and arr.shape[1] == self.faces_per_cell
        wp.launch(kernel=self._massflux_to_array,dim = [cells.shape[0],self.faces_per_cell],inputs = [cells,arr])
