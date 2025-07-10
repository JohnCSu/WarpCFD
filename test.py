
from warp_cfd.FV.finiteVolume import FVM
import numpy as np
from warp_cfd.FV.terms.convection import ConvectionTerm
from warp_cfd.FV.terms.diffusion import DiffusionTerm
from warp_cfd.FV.terms.field import Field
import warp as wp
wp.init()
if __name__ == '__main__':
    from grid import create_hex_grid
    # wp.clear_kernel_cache()
    n =51
    w,l = 1.,1.
    G,nu = 1,1/100
    m = create_hex_grid(n,n,1,(w/n,l/n,0.1))

    # IC = np.load(f'benchmark_n{n}.npy')
    m.set_boundary_value('+X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-X',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('-Y',u = 0,v = 0,w = 0) # No Slip
    m.set_boundary_value('+Y',u = 1,v = 0,w = 0) # Velocity Inlet

    '''
    Add check that All bf have some fixed value => Boundary IDs should equal same length as boundary faces
    '''
    m.set_gradient_value('-Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+Z',u=0,v=0,w=0,p=0) # No penetration condition
    m.set_gradient_value('+X',p = 0) # No Slip
    m.set_gradient_value('-X',p = 0) # No Slip
    # m.set_gradient_value('-Y',u=0,v=0,w=0) # No Slip
    m.set_gradient_value('+Y',p = 0) # Velocity Inlet
    m.set_gradient_value('-Y',p = 0) # Velocity Inlet
    
    m.set_cell_value(0,p= 0)
    
    model = FVM(m,density = 1.,viscosity= nu)
    model.init_step()

    results = m.pyvista_mesh
    
    convection = ConvectionTerm(model,model.fields[0:-1]) # We only want velocities
    diffusion = DiffusionTerm(model,model.fields[0:-1])

    p_correction = DiffusionTerm(model,model.fields[-1])

    np.set_printoptions(linewidth=500,threshold=1e10,precision = 7)
    model.MAX_STEPS = 1
    for i in range(model.MAX_STEPS):
        model.face_interpolation()
        model.calculate_gradients()
        model.calculate_mass_flux(rhie_chow=True)

        convection(model)
        diffusion(model,viscosity = nu)
    # exit()
    # print(model.mass_fluxes.numpy())