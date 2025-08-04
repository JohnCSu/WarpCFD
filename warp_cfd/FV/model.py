import warp as wp
import numpy as np

from warp import sparse
from warp.types import vector
from warp_cfd.FV.utils import to_vector_array

from warp_cfd.preprocess import Mesh
import warp_cfd.FV.mesh_structs as Cells
from warp_cfd.FV.boundary.conditions import apply_BC_kernel,set_initial_conditions_kernel
from warp_cfd.FV.convergence import Convergence
from warp_cfd.FV.field import Field
from warp_cfd.FV.Ops import model_ops
from warp_cfd.FV.interpolation_Schemes import boundary_calculate_face_interpolation_kernel,internal_calculate_face_interpolation_kernel,linear_interpolation,upwind
from typing import Any

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
        self.vector3d_dtype = vector(3,dtype=self.float_dtype)
        self.cellType = mesh.cellType
        self.mesh = mesh
        assert mesh.num_outputs == len(output_variables), 'Number of defined output variables must match number of outputs given in mesh'
        self.settings = CFD_settings()
        self.Convergence = Convergence()
        self.skew_correction = True
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

        self.cell_struct,self.face_struct,self.node_struct= Cells.create_mesh_structs(self.nodes_per_cell,self.faces_per_cell,self.nodes_per_face,self.dimension,self.float_dtype,self.int_dtype)
        self.cells = wp.zeros(shape=self.num_cells,dtype= self.cell_struct)
        '''Array of cell structs for storing information about cell dependent values'''
        self.faces = wp.zeros(shape = self.num_faces,dtype=self.face_struct)
        '''Array of face structs for storing information about face dependent values'''
        self.nodes = wp.zeros(shape= self.num_nodes,dtype=self.node_struct)
        '''Array of node structs for storing inforation about nodes (mainly id and coordinates)'''

        
        self.reference_pressure_cell_id = None
        self.reference_pressure = 0.


        self.init_step()

    def init_step(self):
        Cells.init_structs(self.cells,self.faces,self.nodes,self.cell_properties,self.face_properties,self.node_properties,float_dtype= self.float_dtype)
        self.boundary_face_interpolation = boundary_calculate_face_interpolation_kernel(self.cell_struct,self.face_struct,self.float_dtype)
        self.internal_face_interpolation = internal_calculate_face_interpolation_kernel(linear_interpolation,self.cell_struct,self.face_struct,self.skew_correction,self.float_dtype)
        self.internal_face_interpolation_upwind = internal_calculate_face_interpolation_kernel(upwind,self.cell_struct,self.face_struct,self.skew_correction,self.float_dtype)         
        self.init_global_arrays()
    
    def set_initial_conditions(self,IC:wp.array,output_indices = None):
        cell_values = self.cell_values
        if output_indices is None:
            output_indices = wp.array([i for i in range(IC.shape[-1])],dtype = self.int_dtype)
        wp.launch(kernel= set_initial_conditions_kernel, dim = [cell_values.shape[0],len(output_indices)],inputs= [IC,cell_values,output_indices])
        # self.mesh_ops.set_initial_conditions(IC,self.cell_values)
        # wp.launch(,)

    def set_reference_pressure(self,cell_id:int,value:float = 0.):
        self.reference_pressure_cell_id = cell_id
        self.reference_pressure = float(value)

    def init_global_arrays(self):
        
        self.boundary_value = self.face_properties.boundary_value
        self.boundary_type = self.face_properties.boundary_type
        self.boundary_ids = self.face_properties.boundary_face_ids

        self.cell_values = wp.zeros((self.num_cells,self.num_outputs),dtype = self.float_dtype)
        self.cell_gradients = wp.zeros(shape = (self.num_cells,self.num_outputs),dtype = self.vector3d_dtype)
        self.face_values = wp.zeros(shape = (self.num_faces,self.num_outputs),dtype= self.float_dtype)
        self.face_gradients = wp.zeros(shape = (self.num_faces,self.num_outputs),dtype= self.float_dtype)
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
            raise ValueError('Only cell or face struct members can be returned')
        keys = list(struct_.vars.keys())
        index = keys.index(member)

        member_arr = [a[index] for a in arr]
        return np.array(member_arr)
    

    def get_output_indices(self,output_indices: tuple[int|str] |list[int|str] | wp.array[int] | str | int | None):
        if output_indices is None:
            return self.output_indices  
        if isinstance(output_indices,(int)):
            output_indices = [output_indices]
        elif isinstance(output_indices,str):
            assert output_indices in self.output_variables, 'output indices type str was not found in model\'s list of output variables'
            output_indices = [self.fields[output_indices].index]
        elif isinstance(output_indices,(list,tuple)) and all(isinstance(o,str) for o in output_indices):
            output_indices = [self.fields[output].index for output in output_indices]

        return wp.array(output_indices,dtype = wp.int32)
            
    def set_boundary_conditions(self):
        # self.mesh_ops.apply_BC(self.boundary_ids,self.boundary_type,self.face_values,self.face_gradients,self.faces)
        wp.launch(apply_BC_kernel,dim = (self.boundary_ids.shape[0],self.num_outputs),inputs =[self.face_values,self.face_gradients,self.boundary_value,self.boundary_ids,self.boundary_type])

    def face_interpolation(self,output_indices: None | list | wp.array = None,upwind = False):
        '''
        Interpolate values to boundary (if von neumann) and internal faces. If output_indices is None, All variables defined in model are interpolated to faces
        '''
        output_indices = self.get_output_indices(output_indices)
        
        if upwind:
            internal_face_interpolation_scheme = self.internal_face_interpolation_upwind
        else:
            internal_face_interpolation_scheme = self.internal_face_interpolation

        wp.launch(kernel = internal_face_interpolation_scheme, dim = (self.face_properties.internal_face_ids.shape[0],output_indices.shape[0]), inputs = [self.cell_values,
                                                                                                                                                        self.cell_gradients,
                                                                                                                                                        self.mass_fluxes,
                                                                                                                                                        self.face_values,
                                                                                                                                                        self.faces,
                                                                                                                                                        self.cells,
                                                                                                                                                        self.face_properties.internal_face_ids,
                                                                                                                                                        output_indices])
        wp.launch(kernel = self.boundary_face_interpolation, dim = (self.boundary_ids.shape[0],output_indices.shape[0]), inputs = [self.cell_values,self.face_values,self.face_gradients,self.faces,self.cells,self.boundary_ids,self.boundary_type,output_indices])
    

    def calculate_gradients(self,output_indices: None | list | wp.array = None):
        '''
        Calculate Cell Centered gradient. If output_indices is None, All variables defined in model are interpolated. See self.get_output_indices for the type of valid inputs
        '''
        output_indices = self.get_output_indices(output_indices)        
        self.cell_gradients.zero_()

        wp.launch(kernel=model_ops.calculate_gradients_kernel,dim = (self.num_cells,self.faces_per_cell,len(output_indices)),inputs = [self.face_values,self.cell_gradients,self.cells,self.faces,self.nodes,output_indices])

    
    def get_gradient(self,output_index: str | int,coeff = 1.):
        '''Return a copy of the gradient of a specific output. Returns a 2D array'''
        
        assert isinstance(output_index,(str,int)), 'output_index can only be int or str'
        if isinstance(output_index,str):
            output_index = self.fields[output_index].index

        gradient_array = wp.array2d(shape = (self.num_cells,3),dtype = self.float_dtype)
        if isinstance(coeff,float):
            coeff = wp.array([coeff],dtype= self.float_dtype)
        wp.launch(model_ops.get_gradient_kernel,dim = (self.num_cells,3),inputs= [gradient_array,coeff,self.cell_gradients,output_index] )
        
        return gradient_array
    
    def relax(self,new_value,alpha,output_index):
        
        assert isinstance(output_index,(str,int)), 'output_index can only be int or str'
        if isinstance(output_index,str):
            output_index = self.fields[output_index].index

        wp.launch(explicit_relax,dim = self.num_cells, inputs = [self.cell_values,new_value,alpha,output_index])

    def calculate_mass_flux(self):
        # self.mesh_ops.calculate_mass_flux(self.mass_fluxes,self.face_values,self.cells,self.faces)
        wp.launch(kernel = model_ops.calculate_mass_flux_kernel,dim = (self.mass_fluxes.shape[0]),inputs = [self.mass_fluxes,self.face_values,self.cells,self.faces,])
       

    def divFlux(self,arr = None):
        if arr is None:
            arr = wp.zeros(shape = self.cells.shape[0],dtype= self.float_dtype)
        arr.zero_()
        # self.mesh_ops.calculate_divergence(self.mass_fluxes,self.cells,arr)
        wp.launch(kernel=model_ops.divFlux_kernel, dim = (self.num_cells,self.faces_per_cell),inputs = [self.mass_fluxes,self.cells,arr])
        return arr
    

    def check_convergence(self,vel_matrix:sparse.BsrMatrix,velocity:wp.array,b:wp.array,div_u=None,velocity_correction:wp.array=None,p_correction:wp.array=None,log = True):
        '''
        Checks the following:
        divergence residual
        momentum equations
        pressure correction residual
        '''
        l2_norm =  lambda Ax,b :np.linalg.norm(Ax-b,ord = 2)
        div_u = self.divFlux(div_u)
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
    

    def replace_cell_values(self,output_indices:int | list | tuple |wp.array,value: wp.array):
        '''
        Completely Override the values stored in the cell values array. To add to the cell values, see `update_cell_values()` method instead
        '''
        output_indices = self.get_output_indices(output_indices)
        # if isinstance(field_idx,(list,tuple,wp.array)):
        assert output_indices.shape[0]*self.num_cells == value.shape[0]
        wp.launch(model_ops.fill_outputs_from_matrix_vector,dim = (self.num_cells,output_indices.shape[0]), inputs =  [value,self.cell_values,output_indices])
        


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


@wp.kernel
def explicit_relax(cell_values:wp.array2d(dtype=Any),new_value:wp.array(dtype=Any),alpha:float,global_output_index:int):
    i = wp.tid()
    cell_values[i,global_output_index] = alpha*new_value[i] + (1.-alpha)*cell_values[i,global_output_index] 

