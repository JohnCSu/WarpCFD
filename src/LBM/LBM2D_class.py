class LBM2D:
    def __init__(self,width:float,height:float,element_length:float,viscosity:float,speed:float,density:float,float_type = ti.f32) -> None:
        self.dimensions = 2
        directions,weights = self.DQ29()
        self.bounding_box = (width,height)
        self.float_type = float_type
        self.shape = (int(width//element_length),int(height//element_length))
        self.offset = (0,0)
        self.fluid_shape = tuple([s for s in self.shape])
        print(self.shape)
        self.f = ti.Vector.field(n = 9,dtype= self.float_type,shape = self.shape,offset = self.offset)
        self.f_next = ti.Vector.field(n = 9,dtype= self.float_type,shape = self.shape,offset = self.offset)
        self.density = ti.field(self.float_type,shape = self.shape,offset = self.offset)
        self.pressure = ti.field(self.float_type,shape = self.shape,offset = self.offset)
        self.velocity = ti.Vector.field(n = 2, dtype= self.float_type, shape = self.shape,offset = self.offset)
        # Boundary Fields
        self.boundary_condition = ti.field(dtype=ti.int8,shape = self.shape,offset = self.offset)
        self.boundary_value = ti.Vector.field(n=2,dtype=self.float_type,shape= self.shape,offset = self.offset)
        self.boundary_to_reflect = ti.Vector.field(n = 9,dtype=ti.int8,shape =self.shape,offset = self.offset)
        self.boundary_reflect_indices = ti.Vector([0,3,4,1,2,7,8,5,6],dt = ti.int8)
        # lattice velocities and weighting
        self.lattice_velocities = ti.Vector.field(n = 2 ,dtype=ti.int8,shape = (9,))
        self.lattice_weights = ti.field(dtype=self.float_type,shape = (9,))
        self.lattice_velocities.from_numpy(directions)
        self.lattice_weights.from_numpy(weights)
        # Physcial Constants
        self.physical_viscosity:float = viscosity
        self.physical_dx = element_length
        self.physical_speed = speed
        self.physical_density = density
        self.physical_dt = self.physical_dx/self.physical_speed
        #Scaling factors
        self.conversion_dt = self.physical_dx/self.physical_speed
        self.conversion_x = self.physical_dx
        self.conversion_speed = (3**0.5)*self.physical_speed
        self.conversion_viscosity = self.physical_speed*self.conversion_x
        self.conversion_density = density
        # Non dimensionalised quantities
        self.dt:float = 1.
        self.dx:float = 1.
        self.cs:float = 1/(3**0.5)

        self.viscosity = self.physical_viscosity/self.conversion_viscosity
        self.tau = self.viscosity/(self.cs**2*self.dt) + 0.5 
        self.Re = self.physical_speed*self.physical_dx/self.physical_viscosity
        self.initialise()

    def DQ29(self):
        directions = np.array([
                [ 0,  0],  # Rest particle
                [ 1,  0],  # Right
                [ 0,  1],  # Up
                [-1,  0],  # Left
                [ 0, -1],  # Down
                [ 1,  1],  # Top-right
                [-1,  1],  # Top-left
                [-1, -1],  # Bottom-left
                [ 1, -1]   # Bottom-right
            ])
        weights = np.array([
            4/9,
            1/9,
            1/9,
            1/9,
            1/9,
            1/36,
            1/36,
            1/36,
            1/36, 
        ])
        return directions,weights
    
    def initialise(self,perturbation = False):
        self.f.fill(1/9)
        self.f_next.fill(1/9)
        
        self.boundary_condition.fill(0)
        self.boundary_value.fill(0)
        self.boundary_to_reflect.fill(0)

        self.density.fill(1.)
        self.velocity.fill(0.)
    def reset_fields(self):
        self.f = self.f_next
        self.f_next = ti.Vector.field(n = 9,dtype= self.float_type,shape = self.shape,offset = self.offset)
