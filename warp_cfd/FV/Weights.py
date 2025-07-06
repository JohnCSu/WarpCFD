import warp as wp

def create_weight_struct(float_dtype = wp.float32):
    @wp.struct
    class Weights:
        owner: float_dtype 
        neighbor: float_dtype
    return Weights



    

