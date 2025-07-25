import warp as wp
import pyvista as pv
import numpy as np
from pyvista import CellType
from warp.types import vector
from warp import sparse

from warp_cfd.FV.Ops.array_ops import add_1D_array,sub_1D_array,inv_1D_array,to_vector_array,mult_scalar_1D_array,absolute_1D_array,power_scalar_1D_array,sum_1D_array,L_norm_1D_array

from warp_cfd.preprocess import Mesh
import warp_cfd.FV.cells as Cells
from warp_cfd.FV.Weights import create_weight_struct
from warp_cfd.FV.Ops import Mesh_Ops,Weights_Ops,Matrix_Ops,Pressure_correction_Ops 
from warp_cfd.FV.convergence import Convergence
from warp_cfd.FV.utils import COO_Arrays
from warp_cfd.FV.deprecated.intermediate_velocity import intermediate_velocity_step
from warp_cfd.FV.deprecated.pressure_correction import pressure_correction_step
from warp_cfd.FV.field import Field



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
    def __init__(self,mesh:Mesh,output_variables = None,density:float = 1000,viscosity:float = 1e-3,float_dtype = wp.float32,int_dtype = wp.int32):
        self.density = density
        self.viscosity = viscosity
        self.gridType:str = mesh.gridType
        self.dimension:int = mesh.dimension
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.cellType = mesh.cellType
        self.mesh = mesh
        assert mesh.num_outputs == len(output_variables), 'Number of defined output variables must match number of outputs given in mesh'
        self.settings = CFD_settings()
        self.Convergence = Convergence()

        self.face_properties = mesh.face_properties.to_NVD_warp(self.float_dtype)
        self.cell_properties = mesh.cell_properties.to_NVD_warp(self.float_dtype)
        self.node_properties = to_vector_array(mesh.nodes,self.float_dtype) 
        # Output for each cell: T,C,4 (u,v,w,p)
        self.input_variables = ['x','y','z','t']
        
        if output_variables is None:
            self.output_variables = ['u','v','w','p']
        else:
            self.output_variables = output_variables

        self.fields = {output:Field(output,i) for i,output in enumerate(self.output_variables)}
        
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

        Ops_args = [self.cell_struct,self.face_struct,self.node_struct,self.weight_struct,self.cell_properties,self.face_properties,self.num_outputs,self.float_dtype,self.int_dtype]

        self.mesh_ops = Mesh_Ops(*Ops_args)
        self.weight_ops = Weights_Ops(*Ops_args) 
        self.matrix_ops = Matrix_Ops(*Ops_args)
        self.pressure_correction_ops = Pressure_correction_Ops(*Ops_args)
    
    def init_step(self):
        Cells.init_structs(self.cells,self.faces,self.nodes,self.cell_properties,self.face_properties,self.node_properties,float_dtype= self.float_dtype)
        self.mesh_ops.init()
        self.weight_ops.init()
        self.matrix_ops.init()
        self.pressure_correction_ops.init()        
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
    
        self.update_cell_values_kernel = update_cell_values(self.float_dtype)
        self.update_cell_values_multi_kernel = update_cell_values_multi(self.float_dtype)
    def struct_member_to_array(self,member = 'mass_fluxes',struct = 'cells'):
        if struct == 'cells':
            struct_ = self.cell_struct
            arr = self.cells.numpy()
        elif struct == 'faces':
            struct_ = self.face_struct
            arr = self.faces.numpy()
        else:
            raise ValueError()
        keys = list(struct_.vars.keys())
        index = keys.index(member)

        member_arr = [a[index] for a in arr]
        return np.array(member_arr)

    def face_interpolation(self):
         # Use it to store stuff
        self.mesh_ops.apply_BC(self.face_values,self.face_gradients,self.faces)
        # self.mesh_ops.apply_cell_value(self.cells)
        self.mesh_ops.calculate_face_interpolation(self.mass_fluxes,self.cell_values,self.face_values,self.face_gradients,self.cells,self.faces,self.p_index,interpolation_method=0)
        self.mesh_ops.calculate_face_interpolation(self.mass_fluxes,self.cell_values,self.face_values,self.face_gradients,self.cells,self.faces,self.vel_indices,interpolation_method=0)
    def calculate_gradients(self):
        self.cell_gradients.zero_()
        self.mesh_ops.calculate_gradients(self.face_values,self.cell_gradients,self.cells,self.faces,self.nodes,self.p_index)
        self.mesh_ops.calculate_gradients(self.face_values,self.cell_gradients,self.cells,self.faces,self.nodes,self.vel_indices)

    def calculate_mass_flux(self,rhie_chow = True):
        if rhie_chow:
            self.mesh_ops.rhie_chow_correction(self.mass_fluxes,self.cell_values,self.face_values,self.cell_gradients,self.D_face,self.cells,self.faces,self.vel_indices)
        else:
            self.mesh_ops.calculate_mass_flux(self.mass_fluxes,self.face_values,self.cells,self.faces)
       

    def calculate_divergence(self,arr = None):
        if arr is None:
            arr = wp.zeros(shape = self.cells.shape[0],dtype= self.float_dtype)
        arr.zero_()
        self.mesh_ops.calculate_divergence(self.mass_fluxes,self.cells,arr)
        return arr
    def check_convergence(self,vel_matrix:sparse.BsrMatrix,velocity:wp.array,b:wp.array,div_u=None,velocity_correction:wp.array=None,p_correction:wp.array=None,log = True):
        '''
        Checks the following:
        divergence residual
        momentum equations
        pressure correction residual
        '''
        l2_norm =  lambda Ax,b :np.linalg.norm(Ax-b,ord = 2)
        div_u = self.mesh_ops.calculate_divergence(self.mass_fluxes,self.cells,div_u)
        Ax:wp.array = sparse.bsr_mv(vel_matrix,velocity)

        Ax = Ax.numpy().reshape(-1,3)
        b = b.numpy().reshape(-1,3)
        
        
        local_err = self.Convergence.local_continuity_error(div_u)
        
        
        convergence = {
            'momentum_x':self.Convergence.check('momentum_x',Ax[:,0],b[:,0]),
            'momentum_y':self.Convergence.check('momentum_y',Ax[:,1],b[:,1]),
            'momentum_z':self.Convergence.check('momentum_z',Ax[:,2],b[:,2]),
            'continuity':self.Convergence.check('continuity',div_u) if div_u is not None else np.nan,
            'local_continuity':self.Convergence.check('local_continuity',local_err),
            'velocity_correction':self.Convergence.check('velocity_correction',velocity_correction) if velocity_correction is not None else np.nan,
            'pressure_correction':self.Convergence.check('pressure_correction',p_correction) if p_correction is not None else np.nan,
        }
        
        
        # print('Cell ID', np.argmax())
        print(convergence)
        if log:
            self.Convergence.log(convergence)
            self.res = np.abs(p_correction.numpy())
        
        if np.isnan(convergence['momentum_x'][0]):
            raise ValueError()
        return self.Convergence.has_converged()
    

    def replace_cell_values(self,field_idx:int | list | tuple |wp.array,value:float | wp.array):
        '''
        Completely Override the values stored in the cell values array. To add to the cell values, see `update_cell_values()` method instead
        '''

        if isinstance(field_idx,int):
            field_idx = [field_idx]
        # if isinstance(field_idx,(list,tuple,wp.array)):
        field_idx = wp.array(field_idx,dtype=wp.int32)

        self.matrix_ops.fill_outputs_from_matrix_vector(self.cell_values,value,field_idx)


    def update_cell_values(self,field_idx:int | list | tuple |wp.array,value:float | wp.array,scale : float = 1.):
        '''
        add the value to the cell values array. To completely replace the cell values, see `replace_cell_values()` method instead
        '''
        # if isinstance(field_idx,int):
        #     field_idx = [field_idx]

        if isinstance(field_idx,(list,tuple,wp.array)):
            field_idx = wp.array(field_idx,dtype=wp.int32) 
            if isinstance(value,float):
                v = wp.empty(shape=(1,1),dtype= self.float_dtype)
                v.fill_(value)
                value = v
                # value = wp.array([value],dtype= self.float_dtype)

            wp.launch(kernel=self.update_cell_values_multi_kernel,dim = self.num_cells, inputs = [field_idx,scale,value,self.cell_values])

        else:
            if isinstance(value,float):
                value = wp.array([value],dtype= self.float_dtype)    
            wp.launch(kernel=self.update_cell_values_kernel,dim = self.num_cells, inputs = [field_idx,scale,value,self.cell_values])


def update_cell_values(float_dtype):
    @wp.kernel
    def _update_cell_values(field_idx:int,scale:float_dtype,field_value:wp.array(dtype=float_dtype),cell_values:wp.array2d(dtype = float_dtype)):
        i = wp.tid() # C
        
        if field_value.shape[0] == 1:
            value = field_value[0]
        else:
            value = field_value[i]

        wp.atomic_add(cell_values,i,field_idx,scale*value)

    return _update_cell_values


def update_cell_values_multi(float_dtype):
    @wp.kernel
    def _update_cell_values_multi(output_idx:wp.array(dtype=int),scale:float_dtype,field_value:wp.array2d(dtype=float_dtype),cell_values:wp.array2d(dtype = float_dtype)):
        i,o = wp.tid() # C,O
        
        if field_value.shape[0] == 1:
            value = field_value[0,0]
        else:
            value = field_value[i,o]

        wp.atomic_add(cell_values,i,output_idx[o],scale*value)
    return _update_cell_values_multi


# wp.overload(update_cell_values,[int,wp.float64,wp.float64,wp.array(dtype=wp.float64),wp.array2d(dtype=wp.float64)])
# wp.overload(update_cell_values_multi,[int,wp.float64,wp.float64,wp.array2d(dtype=wp.float64),wp.array2d(dtype=wp.float64)])
