import warp as wp
from typing import Any
from warp_cfd.FV.utils import COO_Arrays
from warp.optim import linear
import warp.sparse as sparse
from warp_cfd.FV.mesh_structs import CELL_DICT
from warp.sparse import _bsr_block_index


@wp.kernel
def calculate_BSR_matrix_indices( rows:wp.array(dtype=int),
                        cols:wp.array(dtype=int),
                        offsets:wp.array(dtype=int),
                        cell_structs: wp.array(dtype=Any),
                        face_structs:wp.array(dtype=Any),
                        num_outputs: int):
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




@wp.func
def get_face_idx(vec1:Any,id:int):
    for i in range(vec1.length):
        if vec1[i] == id:
            return i
    
    return -1


@wp.kernel
def calculate_BSR_values_kernel(bsr_rows:wp.array(dtype=int),
                        bsr_columns:wp.array(dtype=int),
                        values:wp.array(dtype= Any),
                        cell_structs:wp.array(dtype= Any),
                        weights:wp.array4d(dtype= Any),
                        num_outputs:wp.int32,
                        scale:Any):
    i = wp.tid()

    row = bsr_rows[i]
    col = bsr_columns[i]
    output_idx = wp.mod(row,num_outputs)
    # output = output_indices[output_idx]
    output = output_idx
    cell_id = row//num_outputs
    neighbor_id = col//num_outputs

    owner_cell = cell_structs[cell_id]

    if row == col:
        for face_idx in range(owner_cell.faces.length):
            # if owner_cell.neighbors[face_idx] != -1: # So not boundary
            wp.atomic_add(values,i, values.dtype(scale)*weights[cell_id,face_idx,output,0])
    else: # Off Diagonal
        face_idx = get_face_idx(owner_cell.neighbors,neighbor_id)
        if face_idx != -1:    
            wp.atomic_add(values,i, values.dtype(scale)*weights[cell_id,face_idx,output,1])




@wp.kernel
def calculate_RHS_values_kernel(b:wp.array(dtype=Any),
                                boundary_face_ids:wp.array(dtype= int),
                                face_structs:wp.array(dtype=Any ),
                                weights:wp.array4d(dtype= Any ),
                                num_outputs:wp.int32,
                                scale:Any):
    
    i,output_idx = wp.tid() #Loop through Boundary faces

    face_id = boundary_face_ids[i]
    cell_id = face_structs[face_id].adjacent_cells[0]
    face_idx = face_structs[face_id].cell_face_index[0]
    # output = output_indices[output_idx]
    row = cell_id*num_outputs +output_idx

    # wp.atomic_add(b,row,weights[cell_id,face_idx,output].neighbor -  weights[cell_id,face_idx,output].neighbor )
    wp.atomic_add(b,row,-b.dtype(scale)*weights[cell_id,face_idx,output_idx,2] )


@wp.kernel
def add_Implicit_cell_based_term_kernel(bsr_row_offsets:wp.array(dtype=int),
                        bsr_columns:wp.array(dtype=int),
                        values:wp.array(dtype= Any),
                        b:wp.array(dtype=Any),
                        weights:wp.array3d(dtype= Any),
                        num_outputs:wp.int32,
                        scale:Any):
    row = wp.tid() # Number of rows
    
    diag_nnz_idx = _bsr_block_index(row,row,bsr_row_offsets,bsr_columns) # Find Diagonal Value index in NNZ indices
    
    
    cell_id = row//num_outputs
    output_idx = wp.mod(row,num_outputs)
    
    # We only care about diagonals weight[C,O,2]
    wp.atomic_add(values,diag_nnz_idx,scale*weights[cell_id,output_idx,0])
    wp.atomic_add(b,row,-scale*weights[cell_id,output_idx,1])





@wp.kernel
def implicit_relaxation_kernel(bsr_rows:wp.array(dtype=int),
                        bsr_columns:wp.array(dtype=int),
                        values:wp.array(dtype= Any),
                        b:wp.array(dtype=Any),
                        cell_values:wp.array2d(dtype=Any),
                        alpha:Any,
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
        
        b[row] +=  (b.dtype(1.)-alpha)/alpha*values[i]*cell_value
        values[i] = values[i]/alpha
        

@wp.kernel
def replace_row_kernel(bsr_row_offsets:wp.array(dtype= int),
                        bsr_columns: wp.array(dtype= int),
                        bsr_values:wp.array(dtype=Any),
                        b:wp.array(dtype=Any),
                        row_ids:wp.array(dtype=int),
                        rhs_values:wp.array(dtype=Any)):
    
    i = wp.tid() # We go through all Cells with fixed pressure values
    
    #For now only for pressure so cell_id = row
    row_id = row_ids[i]
    col_start,col_end = bsr_row_offsets[row_id], bsr_row_offsets[row_id+1]
    for nnz_idx in range(col_start,col_end):
        column = bsr_columns[nnz_idx]
        if row_id == column: #Diagonal
            bsr_values[nnz_idx] = bsr_values.dtype(1.)
        else: # Off diagonal
            bsr_values[nnz_idx] = bsr_values.dtype(0.)
    
    b[row_id] = rhs_values[row_id] # Replace the row with the specified value


@wp.kernel
def get_diagonal_indices_kernel(bsr_row_offsets:wp.array(dtype=wp.int32),
                                bsr_columns:wp.array(dtype=wp.int32),
                                diagonal_index:wp.array(dtype=wp.int32)):
    row = wp.tid()
    diagonal_index[row] = _bsr_block_index(row,row,bsr_row_offsets,bsr_columns) 





'''Overloads FOR FP ONLY'''

for T in [wp.float32,wp.float64]:
    wp.overload(implicit_relaxation_kernel,{"values":wp.array(dtype=T),
                                            "b":wp.array(dtype=T),
                                        "cell_values":wp.array2d(dtype=T),
                                        "alpha":T
                                         })
    
    wp.overload(replace_row_kernel,{"bsr_values":wp.array(dtype=T),
                                            "b":wp.array(dtype=T),
                                        "rhs_values":wp.array(dtype=T),
                                         })
    




'''OVERLOADS WITH CELL STRUCTS'''

for key,(cell_struct,face_struct,node_struct) in CELL_DICT.items():
    float_type = wp.float32 if 'F32' in key else wp.float64
    wp.overload(calculate_BSR_values_kernel,{"cell_structs":wp.array(dtype=cell_struct),
                                    "values":wp.array(dtype= float_type),
                                    "weights":wp.array4d(dtype= float_type),
                                    "scale":float_type
                                         })
    wp.overload(calculate_RHS_values_kernel,{   "face_structs":wp.array(dtype=face_struct),
                                                "b":wp.array(dtype= float_type),
                                                "weights":wp.array4d(dtype= float_type),
                                                "scale":float_type
                                                    })
    
    wp.overload(calculate_BSR_matrix_indices,{"cell_structs":wp.array(dtype=cell_struct),
                                    "face_structs":wp.array(dtype=face_struct),
                                        })
