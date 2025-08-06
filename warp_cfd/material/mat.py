class Material():
    def __init__(self):
        pass

    def property_exists(self,variable:str):
        if not hasattr(self,f'_{variable}'):
            raise ValueError(f'{variable} has not been defined')
    @property
    def density(self):
        self.property_exists('density')
        return self._density
    
    @property
    def viscosity(self):
        '''Dynamic Viscosity'''
        self.property_exists('viscosity')
        return self._viscosity
    
    def set_density(self,density,incompressible = True):
        self.density_is_constant = incompressible
        self._density = density

    def set_viscosity(self,viscosity,newtonian = True):
        self.viscosity_type = 'newtonian' if newtonian else 'non-newtonian'
        self._viscosity = viscosity

    def create_incompressible_newtonian_fluid(self,viscosity,density):
        self.set_density(density) 
        self.set_viscosity(viscosity)