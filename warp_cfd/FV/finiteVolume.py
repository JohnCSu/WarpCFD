import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp.types import vector
from warp import sparse
from warp.optim import linear
from typing import Any

import scipy.sparse as sp_sparse
from warp_cfd.FV.Ops.array_ops import add_1D_array,sub_1D_array,inv_1D_array,to_vector_array,mult_scalar_1D_array,absolute_1D_array,power_scalar_1D_array,sum_1D_array,L_norm_1D_array

from matplotlib import pyplot as plt
from warp_cfd.preprocess import Mesh
import warp_cfd.FV.Cells as Cells
from warp_cfd.FV.Weights import create_weight_struct
from warp_cfd.FV.Ops import Mesh_Ops,Weights_Ops,Matrix_Ops,Pressure_correction_Ops 
from warp_cfd.FV.Convergence import Convergence
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.intermediate_velocity import intermediate_velocity_step
from warp_cfd.FV.pressure_correction import pressure_correction_step
from warp_cfd.FV.terms.field import Field

wp.config.mode = "debug"

# wp.config.verify_fp = True
'''
TODO:
- Form Sparse Matrix
- implement simple Algo
TO ADD LATER:
- Add non-orthoganality for laplacian (Currently no non-ortho)
- Add upwind for convection and options (currently only central differencing)
- Add least squares and node based gauss for gradient calcs

FUTURE
- Time Dependent NS
- Add RANS
- Add LES

What we need from the FVM Class:
 - Interpolate Variables to faces 
 - calc gradients  
 - mass flux calculations
'''
wp.init()
class CFD_settings:
    steady_state: bool = True
    velocity_face_interpolation: int = 'central difference'
    
    interpolation_schemes = {
        'central difference': 0,
        'upwind': 1,
        'upwind-linear':2,
    }


class FVM():
    '''
    Base Class that takes in class Mesh and creates the appropriate object to form terms
    '''
    def __init__(self,mesh:Mesh,density:float = 1000,viscosity:float = 1e-3,dtype = wp.float32,int_dtype = wp.int32):
        self.density = density
        self.viscosity = viscosity
        self.gridType:str = mesh.gridType
        self.dimension:int = mesh.dimension
        self.float_dtype = dtype
        self.int_dtype = int_dtype
        self.cellType = mesh.cellType

        self.settings = CFD_settings()
        self.Convergence = Convergence()

        self.face_properties = mesh.face_properties.to_NVD_warp()
        self.cell_properties = mesh.cell_properties.to_NVD_warp()
        self.node_properties = to_vector_array(mesh.nodes)        
        # Output for each cell: T,C,4 (u,v,w,p)
        self.input_variables = ['x','y','z','t']
        self.output_variables = ['u','v','w','p']
        self.fields = [Field(output,i) for i,output in enumerate(self.output_variables)]
        self.vars_mapping = {var:i for i,var in enumerate(self.output_variables)}
        self.output_indices = wp.array(np.arange(len(self.output_variables)),dtype=self.int_dtype)
        self.vel_indices = wp.array([0,1,2],dtype=self.int_dtype)
        self.p_index = wp.array([3],dtype=self.int_dtype)
        self.num_outputs = len(self.output_variables)

        self.num_nodes = mesh.nodes.shape[0]
        self.num_cells = mesh.cells.shape[0]
        self.num_faces = mesh.face_properties.unique_faces.shape[0]
        
        self.nodes_per_cell = mesh.cells.shape[-1]
        self.faces_per_cell = self.cell_properties.faces_per_cell
        self.nodes_per_face = self.cell_properties.nodes_per_face

        self.cell_struct,self.face_struct,self.node_struct= Cells.create_mesh_structs(self.nodes_per_cell,self.faces_per_cell,self.nodes_per_face,self.num_outputs,self.dimension,self.float_dtype,self.int_dtype)
        self.weight_struct = create_weight_struct(self.float_dtype)
        self.cells = wp.zeros(shape=self.num_cells,dtype= self.cell_struct)
        '''Array of cell structs for storing information about cell dependent values'''
        self.faces = wp.zeros(shape = self.num_faces,dtype=self.face_struct)
        '''Array of face structs for storing information about face dependent values'''
        self.nodes = wp.zeros(shape= self.num_nodes,dtype=self.node_struct)
        '''Array of node structs for storing inforation about nodes (mainly id and coordinates)'''

        self.face_viscosity = None
        
        self.steps = 0
        self.MAX_STEPS = 2

        Ops_args = [self.cell_struct,self.face_struct,self.node_struct,self.weight_struct,self.cell_properties,self.face_properties,self.num_outputs,self.float_dtype,self.int_dtype]

        self.mesh_ops = Mesh_Ops(*Ops_args)
        self.weight_ops = Weights_Ops(*Ops_args) 
        self.matrix_ops = Matrix_Ops(*Ops_args)
        self.pressure_correction_ops = Pressure_correction_Ops(*Ops_args)
    
    def init_step(self):
        '''
        Initialises The Model. For now assume that the mesh and BC are both Fixed in Time
        '''
        Cells.init_structs(self.cells,self.faces,self.nodes,self.cell_properties,self.face_properties,self.node_properties)
        self.mesh_ops.init()
        self.weight_ops.init()
        self.matrix_ops.init()
        self.pressure_correction_ops.init()
        # self.intermediate_velocity_step = intermediate_velocity_step(self.mesh_ops,self.weight_ops,self.matrix_ops)
        # self.pressure_correction_step = pressure_correction_step(self.mesh_ops,self.weight_ops,self.matrix_ops,self.pressure_correction_ops)
        
        # self.intermediate_velocity_step.init(self.cells,self.faces)
        # self.pressure_correction_step.init(self.cells,self.faces)

        self.init_global_arrays()
    

    def set_initial_conditions(self,IC):
        self.mesh_ops.set_initial_conditions(IC,self.cell_values)

    def init_global_arrays(self):
        self.massflux_array = wp.empty(shape = (self.num_cells,self.faces_per_cell))
        
        self.intermediate_velocity = wp.zeros(shape = self.num_cells*self.dimension,dtype=self.float_dtype)
        self.initial_velocity = wp.zeros_like(self.intermediate_velocity)
        self.corrected_velocity = wp.zeros_like(self.intermediate_velocity)
        
        self.cell_values = wp.zeros((self.num_cells,self.num_outputs),dtype = self.float_dtype)
        self.cell_gradients = wp.zeros(shape = (self.num_cells,self.num_outputs),dtype = self.mesh_ops.vector_type)
        self.face_values = wp.zeros(shape = (self.num_faces,self.num_outputs),dtype= self.float_dtype)
        self.face_gradients = wp.zeros(shape = (self.num_faces,self.num_outputs),dtype= self.float_dtype)
        self.D_face = wp.zeros(shape = (self.num_faces),dtype=self.float_dtype)
        self.D_cell = wp.zeros(shape = self.num_cells,dtype = self.float_dtype)
        self.mass_fluxes = wp.zeros(shape = (self.num_faces),dtype= self.float_dtype)
    
    
    def face_interpolation(self):
         # Use it to store stuff
        self.mesh_ops.apply_BC(self.face_values,self.face_gradients,self.faces)
        # self.mesh_ops.apply_cell_value(self.cells)
        self.mesh_ops.calculate_face_interpolation(self.mass_fluxes,self.cell_values,self.face_values,self.face_gradients,self.cells,self.faces,self.output_indices,interpolation_method=0)

    def calculate_gradients(self):
        self.cell_gradients.zero_()
        self.mesh_ops.calculate_gradients(self.face_values,self.cell_gradients,self.cells,self.faces,self.nodes,self.p_index)
        self.mesh_ops.calculate_gradients(self.face_values,self.cell_gradients,self.cells,self.faces,self.nodes,self.vel_indices)

    def calculate_mass_flux(self,rhie_chow = True):
        if rhie_chow:
            self.mesh_ops.rhie_chow_correction(self.mass_fluxes,self.cell_values,self.face_values,self.cell_gradients,self.D_face,self.cells,self.faces,self.vel_indices)
        else:
            self.mesh_ops.calculate_mass_flux(self.mass_fluxes,self.face_values,self.cells,self.faces)
       

