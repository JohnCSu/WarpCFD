import taichi as ti
import numpy as np
# ti.init(arch =ti.cpu)
ti.init(arch=ti.cuda)
# ti.init(debug=True)
from LBM2D_class import LBM2D


LBM = LBM2D(1,1,1/256,viscosity=1.e-5,speed = 0.1,density= 1000,float_type=ti.f64)
LBM.tau,LBM.viscosity

@ti.func
def set_reflection(to_reflect,boundary,opposite_indices):
    for k in ti.static(range(len(to_reflect))):
        boundary[to_reflect[k]] = opposite_indices[to_reflect[k]]
    return boundary

@ti.kernel
def set_walls_BC():
    shape = LBM.fluid_shape
    opposites = LBM.boundary_reflect_indices
    for i,j in ti.ndrange(*LBM.fluid_shape):
        if (i ==0) or (j == 0) or (i ==(shape[0]-1)): # or (j == (shape[-1]-1))  :
            LBM.boundary_condition[i,j] = 1
            #Left and right Wall
            if i == 0: # Left Wall
                to_reflect = [3,7,6]
                LBM.boundary_to_reflect[i,j] = set_reflection(to_reflect,LBM.boundary_to_reflect[i,j],opposites)
            # Right Wall
            elif i == (shape[0]-1):
                to_reflect = [5,1,8]
                LBM.boundary_to_reflect[i,j] = set_reflection(to_reflect,LBM.boundary_to_reflect[i,j],opposites)
            # Bottom Wall
            if j == 0: 
                to_reflect = [7,4,8]
                LBM.boundary_to_reflect[i,j] = set_reflection(to_reflect,LBM.boundary_to_reflect[i,j],opposites)
        # Top Wall
        elif j == (shape[-1]-1):
            LBM.boundary_condition[i,j] = 2
            LBM.boundary_value[i,j] = ti.math.vec2(0.1,0.)
            to_reflect = [6,2,5]
            LBM.boundary_to_reflect[i,j] = set_reflection(to_reflect,LBM.boundary_to_reflect[i,j],opposites)
            
@ti.kernel
def collision():
    for i,j in ti.ndrange(*LBM.fluid_shape):
        density = LBM.density[i,j]
        velocity = LBM.velocity[i,j]
        for k in ti.static(range(LBM.f.n)):
            term_1 = LBM.lattice_velocities[k].dot(velocity)/LBM.cs**2
            term_2 = (term_1**2)/2
            term_3 =  velocity.dot(velocity)/(2*LBM.cs**2)
            f_eq = LBM.lattice_weights[k]*density*(1+ term_1 + term_2 - term_3)
            LBM.f[i,j][k] = LBM.f[i,j][k] - (1/LBM.tau)*(LBM.f[i,j][k] - f_eq)

@ti.kernel
def streaming():
    for i,j in ti.ndrange(*LBM.fluid_shape):
        for k in ti.static(range(LBM.f.n)):
            vel_dir = LBM.lattice_velocities[k]
            i_stream = int(i - vel_dir[0]) 
            j_stream = int(j - vel_dir[1]) 
            
            if (i_stream > -1) and (i_stream < LBM.shape[0]) and ( j_stream > -1) and (j_stream < LBM.shape[1]):    
                f_stream = LBM.f[i_stream,j_stream]
                LBM.f_next[i,j][k] = f_stream[k]  # Pull from neighbors
            else:
                # This is for directions stemming from corners. We just update the f_next[i.j][k] with the same f after collision
                LBM.f_next[i, j][k] = LBM.f[i, j][k]

@ti.func
def halfway_bounceback(f_vec,bc_reflections):
    for k in ti.static(range(f_vec.n)):
        if bc_reflections[k] > 0:
            reflect_idx = bc_reflections[k]
            f_vec[reflect_idx] = f_vec[k]
    return f_vec

@ti.func
def wall_density(f_vec,bc_reflections):
    density:ti.f64 = 0.
    for k in ti.static(range(f_vec.n)):
        if bc_reflections[k] == 0:
            density += f_vec[k]
    return density

@ti.func
def halfway_bounceback_moving_wall(f_vec,bc_reflections,wall_velocity,lattice_weights,lattice_velocities,cs,density):
    for k in ti.static(range(f_vec.n)):
        if bc_reflections[k] > 0:
            reflect_idx = bc_reflections[k]
            
            df = 2*lattice_weights[k]*density*(wall_velocity.dot(lattice_velocities[k]))/(cs**2)
            f_vec[reflect_idx] = f_vec[k] - df
    return f_vec

@ti.kernel
def apply_Boundary_Condition():
    for i,j in ti.ndrange(*LBM.fluid_shape):
        # No Slip
        reflect_vec =LBM.boundary_to_reflect[i,j] 
        if LBM.boundary_condition[i,j] == 1 :        
            LBM.f_next[i,j] = halfway_bounceback(LBM.f_next[i,j],reflect_vec)    
        # Moving Wall
        elif LBM.boundary_condition[i,j] == 2:
            u = LBM.boundary_value[i,j]
            density = LBM.density[i,j]
            LBM.f_next[i,j] = halfway_bounceback_moving_wall(LBM.f_next[i,j],reflect_vec,u,LBM.lattice_weights,LBM.lattice_velocities,LBM.cs,density)
           
@ti.func
def compute_velocity(f_vec,density,lattice_velocities):
    vec = ti.math.vec2(0.,0.)
    for k in range(f_vec.n):
        vec += f_vec[k]*lattice_velocities[k]
    return vec/density

@ti.kernel
def compute_macroscopic():
    for i,j in ti.ndrange(*LBM.fluid_shape):
        f_vec = LBM.f_next[i,j]
        LBM.density[i,j] = f_vec.sum()
        LBM.velocity[i,j] = compute_velocity(f_vec,LBM.density[i,j],LBM.lattice_velocities)

import matplotlib.cm as cm
gui = ti.GUI(res= LBM.shape)

set_walls_BC()
rho_drift = 0
print(LBM.tau,LBM.Re)
for t in range(100_000):
    rho_old = LBM.density.to_numpy().mean()
    collision()
    streaming()
    apply_Boundary_Condition()
    compute_macroscopic()
    LBM.f.copy_from(LBM.f_next)

    if (t % 100) == 0:
        rho = LBM.density.to_numpy()
        rho_drift = rho.mean()-rho_old
        print(f'density = {rho.mean()} drift = {rho_drift}')

        field_np  = LBM.velocity.to_numpy()[:,:]

        u,v = field_np[:,:,0],field_np[:,:,1]
        field_np = np.sqrt(u**2+v**2)
        # field_norm = (field_np - field_np.min()) / (field_np.max() - field_np.min())
        field_norm = (field_np--0)/(0.1 - (-0))
        colored_image = cm.jet(field_norm)[:, :, :3]
        gui.set_image(colored_image)
        gui.show()
    