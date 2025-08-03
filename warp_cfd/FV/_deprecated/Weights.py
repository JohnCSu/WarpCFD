import warp as wp
from typing import Any

def create_weight_struct(float_dtype = wp.float32, implicit = True):
    if implicit:
        @wp.struct
        class Implicit_Weight:
            owner: float_dtype 
            neighbor: float_dtype
            explicit_term:float_dtype

        return Implicit_Weight
    else:
        @wp.struct
        class Explicit_Weight:
            explicit_term:float_dtype
        return Explicit_Weight
        



    

