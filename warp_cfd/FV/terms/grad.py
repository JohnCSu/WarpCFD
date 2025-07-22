from warp_cfd.FV.model import FVM
from warp_cfd.FV.field import Field
from warp_cfd.FV.Implicit_Schemes.gradient_interpolation import central_difference
from warp_cfd.FV.terms.terms import Term
from typing import Any
import warp as wp

class GradTerm(Term):
    def __init__(self, fv: FVM, field: Field ) -> None:
        assert isinstance(field, Field), 'grad Term can only be in terms of 1 field only'
        super().__init__(fv, field, implicit= False, need_global_index= True)

        self.weights = wp.zeros(shape= fv.num_cells*fv.dimension,dtype= self.float_dtype)

        self.get_gradTerm = create_gradTerm(fv.cell_struct,fv.dimension)

    def calculate_weights(self, fv: FVM, **kwargs: Any) -> Any:
        wp.launch(kernel=self.get_gradTerm,dim = (fv.num_cells,fv.dimension),inputs = [self.weights,fv.cell_gradients,fv.cells,self.fields[0].index])


def create_gradTerm(cell_struct,dimension):
    @wp.kernel
    def get_gradTerm(weights:wp.array(dtype= float),cell_gradients:wp.array2d(dtype= Any),cell_structs:wp.array(dtype=cell_struct),global_output_var:int):
        i,dim = wp.tid() # Go by C,dim
        
        row = i*dimension + dim

        weights[row] =  (cell_gradients[i,global_output_var][dim])*cell_structs[i].volume

    return get_gradTerm

