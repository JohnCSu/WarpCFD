import warp as wp
from warp_cfd.FV.model import FVM
from warp_cfd.FV.utils import COO_Arrays
import warp.sparse as sparse
from warp_cfd.FV.Ops.matrix_ops import Matrix_Ops
from warp_cfd.FV.terms.terms import Term
from warp_cfd.FV.field import Field
from warp.optim import linear
from warp_cfd.FV.utils import bsr_to_coo_array
from warp_cfd.FV.utils import green_gauss_gradient

class Matrix():
    def __init__(self,fvm:FVM,fields:str| list[str],solver = linear.bicgstab) -> None:
        if isinstance(fields,str):
            fields = [fields]

        need_globals = True
        for f in fields:
            if f not in fvm.output_variables:
                need_globals = False

        if need_globals:
            self.fields = [fvm.fields[f] for f in fields]
            self.global_output_indices = wp.array([f.index for f in self.fields],dtype= wp.int32)
        else:
            self.fields = [Field(f) for f in fields]
            self.global_output_indices = None

        self.num_outputs =len(fields)
        
        self.COO_array = COO_Arrays(fvm.cell_properties.nnz_per_cell,self.num_outputs,fvm.float_dtype,fvm.int_dtype)
        # WE mak A*u = B matrix A. In 3D we have 3
        self.sparse_rows = fvm.num_cells*self.num_outputs
        

        self.linear_solver = solver
        self.max_iter = 500
        if fvm.float_dtype == wp.float32:
            self.tol = 1.e-6
        elif fvm.float_dtype == wp.float64:
            self.tol = 1.e-9

        self.initial = True
        
        
        self.rhs = wp.zeros(shape=(self.sparse_rows),dtype=fvm.float_dtype)

        fvm.matrix_ops.calculate_BSR_matrix_indices(self.COO_array,fvm.cells,fvm.faces,num_outputs = self.num_outputs)
        self.A = sparse.bsr_from_triplets(self.sparse_rows,self.sparse_rows,
                                                      rows = self.COO_array.rows,
                                                      columns = self.COO_array.cols,
                                                      values = self.COO_array.values,
                                                      prune_numerical_zeros= False)
        
        self.rows = self.A.uncompress_rows()
        self._diagonal = wp.zeros_like(self.x)
        self.matrix_ops = fvm.matrix_ops
        self.float_dtype = fvm.float_dtype
        self.int_dtype = fvm.int_dtype
        self.fvm = fvm
        
        self.initial_gradient = False

    @property
    def x(self):
        if hasattr(self,'_x') is False:
            self._x = wp.zeros_like(self.rhs)
        return self._x

    @property
    def output_gradient(self):
        if not hasattr(self,'_output_gradient'):
            self._output_gradient = wp.zeros(shape = (self.fvm.num_cells,self.fvm.dimension),dtype=self.float_dtype)

        return self._output_gradient 
    @property
    def diagonal(self):
        '''Return Diagonal Of Sparse Matrix'''
        # return sparse.bsr_get_diag(self.A,self._diagonal)
        return sparse.bsr_get_diag(self.A)
    

    @property
    def matrix(self):
        '''
        Return the sparse CSR Matrix that will be solved
        '''
        return self.A
    
    @property
    def dense(self):
        '''
        Return the dense representation of the matrix to be solved. Note this scales N^2 to the number of cells so should primarily be used for debugging purposes on small
        '''
        return bsr_to_coo_array(self.A).toarray()

    def form_system(self,implicit_terms:list[Term] = None,explicit_terms:list[Term]= None,*,fvm:FVM= None):
        self.reset()
        
        fvm = self.fvm if fvm is None else fvm

        if implicit_terms is not None:
            if isinstance(implicit_terms,Term):
                implicit_terms = [implicit_terms]
        
            for term in implicit_terms:
                assert term.implicit
                assert term.weights.shape[-2] ==  self.num_outputs
                # fvm.matrix_ops.calculate_BSR_matrix(self.A,fvm.cells,weights,output_indices,rows= self.vel_matrix_rows,flip_sign = True)
                self.matrix_ops.calculate_Implicit_LHS(self.A,fvm.cells,term.weights,term.scale,self.rows)
                self.matrix_ops.calculate_Implicit_RHS(self.rhs,fvm.faces,term.weights,term.scale)
        
        if explicit_terms is not None:
            if isinstance(explicit_terms,Term):
                explicit_terms = [explicit_terms]

            for rhs_term in explicit_terms:
                assert not rhs_term.implicit
                self.matrix_ops.add_to_RHS(self.rhs,rhs_term.weights,rhs_term.scale)

    def add_RHS(self,arr:wp.array,scale = 1.):
        self.matrix_ops.add_to_RHS(self.rhs,arr,scale)

    def relax(self,relaxation_factor,fvm:FVM = None):
        fvm = self.fvm if fvm is None else fvm
        assert self.global_output_indices is not None, 'Relaxation is only available for fields defined in fvm'
        self.matrix_ops.implicit_relaxation(self.A,self.rhs,fvm.cell_values,relaxation_factor,self.global_output_indices,self.rows)

    
    def replace_row(self,row_id:int | wp.array,rhs_value: float | wp.array):
        if isinstance(row_id,int):
            row_id= wp.array([row_id],dtype=wp.int32)
        if isinstance(rhs_value,float):
            v = wp.empty(shape=row_id.shape,dtype=self.float_dtype)
            v.fill_(rhs_value)

        assert row_id.shape == v.shape

        self.matrix_ops.replace_row(self.A,self.rhs,row_id,v)
    def solve_Axb(self,x = None):
        if x is None:
            x = self.x
        
        result = self.matrix_ops.solve_Axb(self.A,x,self.rhs,self.linear_solver,self.tol,self.max_iter)
        return result,x
    
    def reset(self):
        self.rhs.zero_()
        self.A.values.zero_()

    def calculate_gradient(self,coeff:float|wp.array = 1.,fvm = None):
        ''' Currently Assumes vonneumann BC !!!! Calculate the gradient of the output using green gauss '''
        fvm = self.fvm if fvm is None else fvm
        assert len(self.fields) == 1, 'Gradient Output Only for matrix related to one field'
        output_gradient = green_gauss_gradient(self.x,self.output_gradient,coeff,fvm)
        return output_gradient