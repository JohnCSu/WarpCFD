import warp as wp
import numpy as np
from typing import Any
'''
Primitive Operations for arrays with numerical types e.g. add, mult, div
'''


@wp.kernel
def add_1D_array_kernel(a:wp.array(dtype = Any),b:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    '''he'''
    i = wp.tid()
    c[i] = a[i] + b[i]



@wp.kernel
def sub_1D_array_kernel(a:wp.array(dtype = Any),b:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    '''he'''
    i = wp.tid()
    c[i] = a[i] - b[i]


@wp.kernel
def mult_1D_array_kernel(a:wp.array(dtype = Any),b:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    '''he'''
    i = wp.tid()
    c[i] = a[i] * b[i]


@wp.kernel
def div_1D_array_kernel(a:wp.array(dtype = Any),b:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    '''he'''
    i = wp.tid()
    c[i] = a[i] / b[i]

@wp.kernel
def inv_1D_array_kernel(a:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    i = wp.tid()
    c[i] = 1./a[i]

@wp.kernel
def mult_scalar_1D_array_kernel(a:wp.array(dtype = Any),scalar: Any,c:wp.array(dtype = Any)):
    i = wp.tid()
    c[i] = scalar*a[i]

@wp.kernel
def add_scalar_1D_array_kernel(a:wp.array(dtype = Any),scalar: Any,c:wp.array(dtype = Any)):
    i = wp.tid()
    c[i] = scalar + a[i]

@wp.kernel
def power_scalar_1D_array_kernel(a:wp.array(dtype = Any),scalar: Any,c:wp.array(dtype = Any)):
    i= wp.tid()
    c[i] = a[i]**scalar

@wp.kernel
def absolute_1D_array_kernel(a:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    i = wp.tid()
    c[i] = wp.abs(a[i])

@wp.kernel
def naive_sum_1D_array_kernel(a:wp.array(dtype = Any),c:wp.array(dtype = Any)):
    i = wp.tid()
    wp.atomic_add(c,0,a[i])

def add_1D_array(a:wp.array,b:wp.array,c:wp.array):
    '''add a and b element wise into array c i.e. `c[i] = a[i] + b[i]`'''
    assert a.shape[0] == b.shape[0] and c.shape[0] == b.shape[0]
    assert len(a.shape) == 1 and len(b.shape) == 1 and len(c.shape) == 1

    wp.launch(kernel=add_1D_array_kernel,dim = a.shape[0],inputs=[a,b,c])
    return c
def sub_1D_array(a:wp.array,b:wp.array,c:wp.array):
    '''subtract b from a element wise into array c i.e. `c[i] = a[i] - b[i]`'''
    assert a.shape[0] == b.shape[0] and c.shape[0] == b.shape[0]
    assert len(a.shape) == 1 and len(b.shape) == 1 and len(c.shape) == 1

    wp.launch(kernel=sub_1D_array_kernel,dim = a.shape[0],inputs=[a,b,c])
    return c
def mult_1D_array(a:wp.array,b:wp.array,c:wp.array):
    '''multiply a and b element wise into array c i.e. `c[i] = a[i] * b[i]`'''
    assert a.shape[0] == b.shape[0] and c.shape[0] == b.shape[0]
    assert len(a.shape) == 1 and len(b.shape) == 1 and len(c.shape) == 1

    wp.launch(kernel=mult_1D_array_kernel,dim = a.shape[0],inputs=[a,b,c])
    return c
def div_1D_array(a:wp.array,b:wp.array,c:wp.array):
    '''divide b from a element wise into array c i.e. `c[i] = a[i] / b[i]`'''
    assert a.shape[0] == b.shape[0] and c.shape[0] == b.shape[0]
    assert len(a.shape) == 1 and len(b.shape) == 1 and len(c.shape) == 1

    wp.launch(kernel=div_1D_array_kernel,dim = a.shape[0],inputs=[a,b,c])
    return c
def inv_1D_array(a:wp.array,b:wp.array):
    '''invert elemens in a into array b i.e. `b[i] = 1/a[i]`'''
    assert a.shape[0] == b.shape[0] and len(a.shape) == 1
    wp.launch(kernel=inv_1D_array_kernel,dim = a.shape[0],inputs=  [a,b])
    return b
def mult_scalar_1D_array(a:wp.array,scalar:float,c:wp.array):
    '''multiply b array with scalar into array c i.e. `c[i] = a[i]*scalar`'''
    assert a.shape[0] == c.shape[0]
    assert len(a.shape) == 1
    wp.launch(kernel=mult_scalar_1D_array_kernel,dim = a.shape[0],inputs=[a,scalar,c])
    return c

def add_scalar_1D_array(a:wp.array,scalar:float,c:wp.array):
    '''add b array with scalar into array c i.e. `c[i] = a[i] + scalar`'''
    assert a.shape[0] == c.shape[0]
    assert len(a.shape) == 1

    wp.launch(kernel=add_scalar_1D_array_kernel,dim = a.shape[0],inputs=[a,scalar,c])
    return c
def power_scalar_1D_array(a:wp.array,scalar:float,b:wp.array):
    assert a.shape[0] == b.shape[0]
    assert len(a.shape) == 1
    wp.launch(kernel=power_scalar_1D_array_kernel,dim = a.shape[0],inputs=[a,scalar,b])
    return b

def absolute_1D_array(a:wp.array,b:wp.array):
    assert a.shape[0] == b.shape[0]
    assert len(a.shape) == 1
    
    wp.launch(kernel=absolute_1D_array_kernel,dim = a.shape[0],inputs=[a,b])
    return b
def sum_1D_array(a:wp.array) -> float: 
    '''Sum up elements in array and return total as float'''
    c = wp.zeros(1,dtype=a.dtype)
    wp.launch(kernel=naive_sum_1D_array_kernel,dim = a.shape[0], inputs = [a])
    return c.numpy()[0]
    

def L_norm_1D_array(a:wp.array,power:float,reduction = None):
    b = wp.empty_like(a)
    wp.copy(dest = b,src = a)
    absolute_1D_array(b,b)
    power_scalar_1D_array(b,power,b)
    if reduction is None:
        return b
    elif reduction == 'sum':
        return sum_1D_array(b)
    else:
        raise ValueError('reduction can be either string "sum" or Nonetype')

def to_vector_array(array:np.ndarray,float_dtype = wp.float32):
    '''
    Convert numpy array to warp array with dtype = vector. For shapes >=2, the last array axis is treated 
    as the vector dimension so a (N,M) numpy array becomes an (N,) warp array with dtype: vector(length=M)
    '''
    # dtype = wp.dtype_from_numpy(array.dtype)
    shape = array.shape
    n = shape[-1]
    vector_type = wp.types.vector(length = n, dtype = float_dtype)
    return wp.array(data = array,dtype=vector_type)
