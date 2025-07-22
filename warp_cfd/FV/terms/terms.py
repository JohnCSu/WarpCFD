import warp as wp

from ..field import Field

from warp_cfd.FV.model import FVM
from typing import Any

class Term:
    weights: wp.array
    scale: float
    global_output_indices: None
    need_global_index:bool
    fields: list[Field]

    def __init__(self,fv:FVM, fields: Field| list[Field],implicit,need_global_index:bool) -> None:
        if isinstance(fields,Field):
            self.fields = [fields]
        elif isinstance(fields,(list,tuple)):
            for f in fields:
                assert isinstance(f,Field) ,'all elements in list or tuple field must be of type Field'
            self.fields = fields
        else:
            raise ValueError('must be type Field or list/tuple of Fields')

        self.field_index = {
            f.name : i for i,f in enumerate(self.fields)
        }

        self.num_outputs = len(self.fields)
        self._implicit = implicit

        if self.implicit:
            self.weights = wp.array(shape=(fv.num_cells,fv.faces_per_cell,self.num_outputs,3),dtype=fv.float_dtype)

        if need_global_index:
            for f in self.fields:
                assert f.name in fv.output_variables , 'A specified field does not exist in FVM object'

            self.global_output_indices = wp.array([f.index for f in self.fields],dtype= wp.int32)

        else:
            self.global_output_indices = wp.empty(shape = 1,dtype= fv.int_dtype)
        self.need_global_index = need_global_index
        
        self.float_dtype = fv.float_dtype
        self.int_dtype = fv.int_dtype

        self.scale:float = 1.
        

    def set_implicit(self,implicit:bool):
        self._implicit = implicit
    @property
    def implicit(self):
        '''
        Boolean on whether term has implicit components in it
        '''
        return self._implicit

    def __pos__(self):
        return self
    
    def __neg__(self) -> bool:
        self.scale = -1.
        return self

    def __call__(self, fv:FVM,*args, **kwargs: Any) -> Any:
        self.calculate_weights(fv,*args,**kwargs )

    def calculate_weights(self,fv:FVM,*args, **kwargs: Any) -> Any:
        pass
    

    
