import warp as wp

from ..field import Field

from warp_cfd.FV.model import FVM
from typing import Any

class Term:
    weights: wp.array
    scale: float
    global_output_indices: wp.array
    need_global_index:bool
    fields: list[Field]

    def __init__(self,fv:FVM, fields: str | list[str],implicit,need_global_index:bool) -> None:
        
        self.fields = []
        if need_global_index: # We need to check that the fields given are in the model
            if isinstance(fields,str):
                fields = [fields]

            if isinstance(fields,(list,tuple)): # This hurts my eyes but it wrokss,..,.,..
                for field_name in fields:
                    if field_name in fv.output_variables:
                        self.fields.append(fv.fields[field_name])
                    else:
                        raise ValueError(f'Field {field_name} was not found in original model only variables {fv.output_variables} are available')
                    
                self.global_output_indices = wp.array([f.index for f in self.fields],dtype= wp.int32)


        else:
            assert isinstance(fields,str), 'Only one field variable can be defined if global index is False'
            self.fields.append(Field(fields,index= -1))
            self.global_output_indices = wp.zeros(1,dtype=wp.int32) 

        self.need_global_index = need_global_index

        self.local_field_index = { # Local field index
            f.name : i for i,f in enumerate(self.fields)
        }

        self.num_outputs = len(self.fields)
        self._implicit = implicit

        if self.implicit:
            self.weights = wp.array(shape=(fv.num_cells,fv.faces_per_cell,self.num_outputs,3),dtype=fv.float_dtype)


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
    

    
