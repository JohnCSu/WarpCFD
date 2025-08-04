import warp as wp
from warp_cfd.FV.model import FVM
from warp_cfd.FV.utils import COO_Arrays
import warp.sparse as sparse
from warp_cfd.FV.terms.terms import Term
from warp_cfd.FV.field import Field
from warp.optim import linear
from warp_cfd.FV.utils import bsr_to_coo_array
import warp_cfd.FV.Ops.matrix_ops as matrix_ops
class Equation():
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

        wp.launch(kernel=matrix_ops.calculate_BSR_matrix_indices,dim = [fvm.num_cells,fvm.faces_per_cell+1,self.num_outputs],inputs = [self.COO_array.rows,
                                                                                                                  self.COO_array.cols,
                                                                                                                  self.COO_array.offsets,
                                                                                                                  fvm.cells,
                                                                                                                  fvm.faces,
                                                                                                                  self.num_outputs
                                                                                                                  ])

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
                self.calculate_Implicit_LHS(self.A,fvm.cells,term.weights,term.scale,self.rows)
                self.calculate_Implicit_RHS(self.rhs,fvm.faces,term.weights,term.scale,fvm.boundary_ids)
        
        if explicit_terms is not None:
            if isinstance(explicit_terms,Term):
                explicit_terms = [explicit_terms]

            for rhs_term in explicit_terms:
                assert not rhs_term.implicit
                self.rhs += rhs_term.scale*rhs_term.weights
                # self.matrix_ops.add_to_RHS(self.rhs,rhs_term.weights,rhs_term.scale)

    @staticmethod
    def calculate_Implicit_LHS(BSR_matrix:sparse.BsrMatrix,cells,weights,scale,rows =None):
        if rows is None:
            rows = BSR_matrix.uncompress_rows()
        assert rows.shape[0] == BSR_matrix.values.shape[0]

        output_indices = wp.array([i for i in range(weights.shape[-2])],dtype= int)
        wp.launch(kernel=matrix_ops.calculate_BSR_values_kernel,dim = BSR_matrix.values.shape[0],inputs=[rows,
                                                                                                BSR_matrix.columns,
                                                                                                BSR_matrix.values,
                                                                                                cells,
                                                                                                weights,
                                                                                                output_indices,
                                                                                                scale])


    @staticmethod
    def calculate_Implicit_RHS(b,faces,weights,scale:float,boundary_ids):
        output_indices = wp.array([i for i in range(weights.shape[-2])],dtype= int) # Num of outputs as it is C,F,O,3
        wp.launch(kernel= matrix_ops._calculate_RHS_values_kernel, dim = (boundary_ids.shape[0],output_indices.shape[0]),inputs = [b,
                                                                                                                        boundary_ids,
                                                                                                                        faces,
                                                                                                                        weights,
                                                                                                                        output_indices,
                                                                                                                        scale,
                ])



    def add_RHS(self,arr:wp.array,scale = 1.):
        self.rhs += scale*arr
        

    def relax(self,relaxation_factor,fvm:FVM = None):
        fvm = self.fvm if fvm is None else fvm
        assert self.global_output_indices is not None, 'Relaxation is only available for fields defined in fvm'


        if self.rows is None:
            self.rows = self.A.uncompress_rows()
        cols,values = self.A.columns,self.A.values
        wp.launch(kernel=matrix_ops.implicit_relaxation_kernel,dim = self.rows.shape[0],inputs = [self.rows,cols,values,self.rhs,fvm.cell_values,relaxation_factor,self.global_output_indices])
        # matrix_ops.implicit_relaxation(self.A,self.rhs,fvm.cell_values,relaxation_factor,self.global_output_indices,self.rows)

    
    def replace_row(self,row_id:int | wp.array,rhs_value: float | wp.array):
        if isinstance(row_id,int):
            row_id= wp.array([row_id],dtype=wp.int32)
        if isinstance(rhs_value,float):
            value = wp.empty(shape=row_id.shape,dtype=self.float_dtype)
            value.fill_(rhs_value)

        assert row_id.shape == value.shape

        wp.launch(matrix_ops.replace_row_kernel,dim = row_id.shape, inputs=[self.A.offsets,self.A.columns,self.A.values,self.rhs,row_id,value])
    def solve_Axb(self,x = None):
        if x is None:
            x = self.x
        M = linear.preconditioner(self.A)
        result =self.linear_solver(A =self.A,M= M,atol = self.tol, b = self.rhs,x = x,maxiter=self.max_iter) 
        return result,x
    
    def reset(self):
        self.rhs.zero_()
        self.A.values.zero_()
