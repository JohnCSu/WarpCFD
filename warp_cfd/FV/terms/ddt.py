from typing import Any
from warp_cfd.FV.model import FVM
from warp_cfd.FV.terms.terms import Term
import warp as wp


class DdtTerm(Term):
    def __init__(self, fv: FVM, fields: str | list[str],time_scheme = 'backwardEuler') -> None:
        super().__init__(fv, fields, implicit = True, need_global_index = True,cell_based = True)


        if time_scheme == 'backwardEuler':
            self.time_scheme = backwards_Euler
        else:
            raise NotImplementedError('Only backwardEuler valid option now')

    def calculate_weights(self, fv: FVM,dt : float, **kwargs: Any) -> Any:        
        wp.launch(self.time_scheme,dim = (fv.num_cells,self.global_output_indices.shape[0]), inputs = [self.float_dtype(dt),fv.cell_values,fv.cells,self.weights,self.global_output_indices])





@wp.kernel
def backwards_Euler(dt:Any,cell_values:wp.array2d(dtype = Any),cell_structs:wp.array(dtype= Any),weights:wp.array3d(dtype = Any),output_indices:wp.array(dtype = int)):
    cell_id,output = wp.tid()
    global_var_idx = output_indices[output]
    

    Vddt = cell_structs[cell_id].volume/dt
    #C,1,O,3
    weights[cell_id,global_var_idx,0] = Vddt # Cell Based so second column is size 1
    weights[cell_id,global_var_idx,1] = -cell_values[cell_id,global_var_idx]*Vddt # Explicit
