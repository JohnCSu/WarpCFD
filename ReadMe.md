# Warp CFD

A Python based 3D incompressible CFD solver written for GPU CFD using [Nvidia Warp](https://github.com/NVIDIA/warp). The aim is to hopefully create a reasonably fast solver that is easy to tinker and make changes without needing to leave python. The code is also designed with Deep learning integration in mind and so results should easily be converted to tensor format/graph based tensor format

# Origins
This code is a fun side project to better understand CFD and how to implement it. A lot of inspiration of how to represent the computations was taken from [FiPy](https://www.ctcms.nist.gov/fipy/). Initially written for Taichi Lang, Warp was used in favour for its current better maintenence and restrictions that help prevent users from shooting themselves in the foot such no implicit fp operations of different types and mandatory type annotations for kernels. 

# Examples:

Steady State Lid Driven Cavity (Re=100):



# RoadMap (In hopeful order)
- Implement SIMPLE loop
- Add Pressure inlet/outlet
- Add And Test Tetra Elements
- Add And Test Wedge Elements
- Implement Skewness corrections


- Add unsteady flow
    - Pseudo Transient
    - PISO
    - PIMPLE

- Rendering Results

- Add RANS Correction
    - k-epsilon
    - k-omega
 
- Add Explicit solver
    - Backpropagation capability

# Far in Future
- Add LES
- LBM implementation
- distributed computing
- adaptive meshing
- mixed precision

- OpenFoam interopability???



# What I won't work on
- Meshing (go to gmsh for that)


# License
