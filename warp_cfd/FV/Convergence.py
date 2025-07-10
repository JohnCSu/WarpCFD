import warp as wp
import numpy as np
from matplotlib import pyplot as plt
from typing import Any


class Criteria:
    def __init__(self,name,eps = 1e-4,tol='relative',ord =1.,is_flux = False) -> None:
        assert tol == 'relative' or tol == 'abs'
        
        self.name = name
        self.eps = eps
        self.tol = tol
        self.ord = ord
        self.is_converged = False
        self.non_zero = 1e-6
    def residual(self,arr:wp.array,rhs:wp.array|float = 0.,ord = None):
        if ord is None:
            ord = self.ord        
        res = np.linalg.norm((arr - rhs),ord)
        return res
    
    def relative_residual(self,arr:wp.array,rhs:wp.array|float,ord = None):
        ord = self.ord if ord is None else ord
        res = self.residual(arr,rhs,ord)

        rhs_norm = np.linalg.norm(rhs,ord)
        if self.non_zero > rhs_norm:
            Warning('Relative tolerance was used but the rhs_norm was less than the specified non-zero') 
        return res/max(np.linalg.norm(rhs,ord),self.non_zero)
    
    def check(self,arr:wp.array|np.ndarray,rhs:wp.array|float|np.ndarray,ord =None):
        
        if isinstance(arr,wp.array):
            arr = arr.numpy()
        if isinstance(rhs,float):
            rhs = np.array([rhs],dtype = arr.dtype)
        elif isinstance(rhs,wp.array):
            rhs = rhs.numpy()
        
        if self.tol == 'abs':
            res =  self.residual(arr,rhs,ord)
        else:
            res =  self.relative_residual(arr,rhs,ord)

        if res < self.eps:
            self.is_converged = True    
        else:
            self.is_converged = False
        return res, self.is_converged

    
class Convergence:
    def __init__(self):
        self.continuity = Criteria('continuity',eps = 1e-4,tol = 'abs')
        self.local_continuity = Criteria('local_continuity',eps = 1e-4,tol = 'abs')
        self.momentum_x = Criteria('momentum_x',ord =2.,eps = 1e-3)
        self.momentum_y = Criteria('momentum_y',ord =2.,eps = 1e-3)
        self.momentum_z = Criteria('momentum_z',ord =2.,eps = 1e-3)
        self.pressure_correction = Criteria('pressure_correction', tol = 'abs',eps = 1e-4,ord = np.inf )
        self.velocity_correction = Criteria('velocity_correction',tol = 'abs',eps = 1e-4 ,ord =  np.inf)
        self.criterias = [self.continuity,self.momentum_x,self.momentum_y,self.momentum_z,self.pressure_correction,self.velocity_correction]
        self.log_list = []
    def check(self,criteria:str,arr,rhs:float|wp.array = 0.,ord = None):
        criteria:Criteria = getattr(self,criteria)
        return criteria.check(arr,rhs,ord)
    
    def has_converged(self):
        for criteria in self.criterias:
            if criteria.is_converged == False:
                return False 
        return True
    def log(self,step:dict):
        self.log_list.append(step)


    def plot_residuals(self):
        x = np.arange(len(self.log_list))
        for criteria in self.criterias:
            name = criteria.name
            y = [step[name][0] for step in self.log_list]
            plt.plot(x,y, label = name)
        # plt.ylim((0,1))
        plt.yscale('log')
        plt.legend()
        plt.show()

    def local_continuity_error(self,div_u,local_err:wp.array|None =None):
        '''
        For Each cell, calculate the abs(mass_flux) of each face
        '''
        # faces_per_cell = cells.dtype.vars['faces'].type._length_
        if local_err is None:
            local_err = wp.zeros_like(div_u)
        else:
            local_err.zero_()
        wp.launch(kernel=continuity_error,dim= (div_u.shape[0]),inputs=[div_u,local_err])
        
        return local_err



@wp.kernel
def continuity_error(div_u:wp.array(dtype=float),local_Err:wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(local_Err,i,wp.abs(div_u[i]))
   