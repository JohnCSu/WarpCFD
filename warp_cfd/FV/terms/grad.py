from warp_cfd.FV.model import FVM
from warp_cfd.FV.field import Field
from warp_cfd.FV.terms.terms import Term
from typing import Any
import warp as wp

class GradTerm(Term):
    def __init__(self, fv: FVM, field: str ) -> None:
        assert isinstance(field,str), 'grad Term can only be in terms of 1 field string name only'
        super().__init__(fv, field, implicit= False, need_global_index= True)

        self.weights = wp.zeros(shape= fv.num_cells*fv.dimension,dtype= self.float_dtype)

        self.get_gradTerm = create_gradTerm(fv.cell_struct,fv.dimension,float_dtype = self.float_dtype)

    def calculate_weights(self, fv: FVM, **kwargs: Any) -> Any:
        wp.launch(kernel=self.get_gradTerm,dim = (fv.num_cells,fv.dimension),inputs = [self.weights,fv.cell_gradients,fv.cells,self.fields[0].index])


def create_gradTerm(cell_struct,dimension,float_dtype):
    @wp.kernel
    def get_gradTerm(weights:wp.array(dtype= float_dtype),cell_gradients:wp.array2d(dtype= Any),cell_structs:wp.array(dtype=cell_struct),global_output_var:int):
        i,dim = wp.tid() # Go by C,dim
        
        row = i*dimension + dim

        weights[row] =  (cell_gradients[i,global_output_var][dim])*cell_structs[i].volume

    return get_gradTerm

