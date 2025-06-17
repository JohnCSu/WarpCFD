'''
Lid Driven Cavity Example


'''

from grid import create_hex_grid
from FiniteVolume import FVM


m = create_hex_grid(10,10,1,(0.1,0.1,0.1))

FV = FVM(mesh=m,density=1,viscosity=0.01)

