class Material():
    def __init__(self):
        pass

    @property
    def density(self):
        return self._density
    
    @property
    def viscosity(self):
        return self._viscosity
    
    def set_density(self,density,incompressible = True):
        self.density_is_constant = incompressible
        self._density = density

    def set_viscosity(self,viscosity,newtonian = True):
        self.viscosity_type = 'newtonian' if newtonian else 'non-newtonian'
        self._viscosity = viscosity