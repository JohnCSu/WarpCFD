import warp as wp
from typing import Any
from warp.types import vector


@wp.kernel
def _calculate_rUA_kernel(D_cell:wp.array(dtype=Any),
                                    Ap:wp.array(dtype=Any),
                                    cell_structs:wp.array(dtype=Any)
                                    ):
    i = wp.tid() # By Cells
    # diagonals for a cell in u,v,w are always the same for isotropic viscosity
    D = 3
    row = i*D
    D_cell[i] = cell_structs[i].volume/Ap[row]




@wp.kernel
def _interpolate_cell_value_to_face_kernel(D_face:wp.array(dtype=Any),
                                    D_cell:wp.array(dtype=Any),
                                    face_structs:wp.array(dtype = Any)):
    face_id = wp.tid() # K
    # D is a C,3 array This is always 3
    owner_cell_id = face_structs[face_id].adjacent_cells[0]
    owner_D = D_cell[owner_cell_id]
    if face_structs[face_id].is_boundary == 1: #(up - ub)/distance *A
        D_face[face_id] = owner_D
    else:
        neighbor_cell_id = face_structs[face_id].adjacent_cells[1]
        neighbor_D = D_cell[neighbor_cell_id]
        
        D_f = face_structs[face_id].norm_distance[0]*owner_D + face_structs[face_id].norm_distance[1]*neighbor_D
        D_face[face_id] = D_f

@wp.kernel
def get_HbyA_kernel(HbyA:wp.array(dtype=Any),rUA:wp.array(dtype=Any),grad_P:wp.array(dtype=Any)):
    i,j = wp.tid() # C,3
    index = i*3 + j

    HbyA[index] +=  grad_P[index]*rUA[i]



@wp.kernel
def divFlux_kernel(face_values:wp.array(dtype=Any),flux_values:wp.array(dtype=Any),cell_structs:wp.array(dtype=Any),face_structs:wp.array(dtype=Any)):
    
    face_id = wp.tid()
    face = face_structs[face_id]
    # i,j = wp.tid() # C,F
    owner_cell = cell_structs[face.adjacent_cells[0]] 
    normal = owner_cell.face_normal[face.cell_face_index[0]]
    area = face_structs[face_id].area
    #Compute dot product
    v = face_values[face_id]
    flux_values[face_id] = wp.dot(v,normal)*area

@wp.kernel
def sumFluxes_kernel(flux_values:wp.array(dtype=Any),cell_values:wp.array(dtype=Any),cell_structs:wp.array(dtype=Any)):
    i,j = wp.tid() # C,F
    # calculate divergence
    face_id = cell_structs[i].faces[j]
    if cell_structs[i].face_sides[j] == 0: # Means cell is the Owner side of face
        wp.atomic_add(cell_values,i,flux_values[face_id])
    else: # means is neighbor side of face so set to -ve
        wp.atomic_add(cell_values,i,-flux_values[face_id])


def divFlux(cell_value:wp.array,model):
    '''
    Calculate Volume integral of divergence of a array of vectors. Returns flux array
    '''
    faces_per_cell = model.faces_per_cell
    faces =model.faces
    cells = model.cells
    
    face_value:wp.array = wp.zeros(shape = faces.shape[0],dtype= vector(3,dtype=cell_value.dtype))
    flux_array:wp.array = wp.zeros(shape= faces.shape[0],dtype= cell_value.dtype)
    cell_array = wp.zeros(shape = cells.shape[0], dtype= cell_value.dtype)

    # faces_per_cell = cells.vars['faces'].type._length_
    
    
    cell_value_as_vec = wp.array(cell_value.reshape( (-1,3)),dtype= vector(3,dtype= cell_value.dtype))
    face_value = interpolate_cell_value_to_face(face_value,cell_value_as_vec,faces)
    wp.launch(divFlux_kernel,dim = faces.shape[0],inputs = [face_value,flux_array,cells,faces])

    wp.launch(sumFluxes_kernel,dim = [cells.shape[0],faces_per_cell],inputs = [flux_array,cell_array,cells])

    return cell_array,face_value



def calculate_rUA(Ap:wp.array,cells:wp.array,out:wp.array | None = None):

    '''
    Calculate rUA =  1/ap = V/Ap where Ap is the diagonal coeffecient of the momentum matrix
    '''
    
    if out is None:
        out = wp.zeros_like(Ap)
    else:
        assert out.shape[0]*3 == Ap.shape[0], 'Ap is diagonal of 3 velocities so should be 3*num_cells in model'
    
    assert len(Ap.shape) == 1
    
    wp.launch(_calculate_rUA_kernel,dim = cells.shape[0],inputs= [out,Ap,cells])
    return out


def interpolate_cell_value_to_face(face_value:wp.array,cell_value:wp.array,faces:wp.array):
    wp.launch(_interpolate_cell_value_to_face_kernel,dim = faces.shape[0],inputs= [face_value,cell_value,faces])
    return face_value

def get_HbyA(HbyA:wp.array(dtype=float),rUA:wp.array(dtype=float),grad_P:wp.array(dtype=float)):
    wp.launch(get_HbyA_kernel,dim = (rUA.shape[0],3),inputs = [HbyA,rUA,grad_P] )
    return HbyA
