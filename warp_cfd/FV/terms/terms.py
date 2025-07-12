import warp as wp

from .field import Field

from warp_cfd.FV.finiteVolume import FVM
from typing import Any

class Term:
    def __init__(self,fv:FVM, field: Field| list[Field]) -> None:
        if isinstance(field,Field):
            self.field = [field]
        elif isinstance(field,(list,tuple)):
            for f in field:
                assert isinstance(f,Field) ,'all elements in list or tuple field must be of type Field'
            self.field = field
        else:
            raise ValueError('must be type Field or list/tuple of Fields')

        self.field_index = {
            f.name : i for i,f in enumerate(self.field)
        }

        self.num_outputs = len(self.field)
        self.weights = wp.array(shape=(fv.num_cells,fv.faces_per_cell,self.num_outputs),dtype=fv.weight_struct)

        self.global_output_indices = wp.array([f.index for f in self.field],dtype= wp.int32)
        self.float_dtype = fv.float_dtype
        self.int_dtype = fv.int_dtype

    def __call__(self, fv:FVM,*args, **kwargs: Any) -> Any:
        self.calculate_weights(fv,*args,**kwargs )

    def calculate_weights(self,fv:FVM,*args, **kwargs: Any) -> Any:
        pass
    

    
