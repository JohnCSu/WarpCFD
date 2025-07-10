import warp as wp
from typing import Any

def create_weight_struct(float_dtype = wp.float32):
    @wp.struct
    class Weight:
        owner: float_dtype 
        neighbor: float_dtype
        explicit_term:float_dtype

    return Weight




    

