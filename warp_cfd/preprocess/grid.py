import gmsh
import sys

import pyvista as pv
import numpy as np

from .mesh import Mesh
def create_2D_grid(origin,nx,ny,dx,dy,dz = 0.1,*,element_type = 'hex',unstructured_wedge = False,display_mesh = False,save = None,gmsh_version2 = False) -> pv.UnstructuredGrid:
    '''
    Create a 3D grid with a single element in the z direction to represent a 2D grid in 3D space returns a pyvista Undstructured Grid.

    This thing was vibe coded so dont ask me how it works just that it does.
    
    Supports
    - Structured Hex and Wedge elements
    - Unstructured wedge elements

    For unstructured wedge elements, the approximate element size is determined by min of (dx/nx and dy/ny)
    
    

    '''
    gmsh.initialize()

    valid_elements = ['wedge','hex','tet']


    element_id = {
        'wedge':6,
        'hex':5,
        'tet':4,
    }

    pyvista_cell_id = {
        'wedge':13,
        'hex':12,
        'tet':10,
    }

    num_nodes_per_cell = {
        'wedge': 6,
        'hex':8,
        'tet': 4

    }
    assert element_type in valid_elements

    gmsh.model.add("2D_Grid")
    # Step 1: Create 2D rectangle surface
    rect = gmsh.model.occ.addRectangle(*origin,dx,dy)
    
    if element_type == 'wedge':
        if unstructured_wedge:
            # Split the rectangular surface into triangles
            # This returns surface tags, which we extrude
            gmsh.model.occ.fragment([(2, rect)], [])  # Ensures it's ready for face ops
            gmsh.model.occ.synchronize()


            # Mesh.CharacteristicLengthMin = 0.01;
            # Get the two triangle surfaces from the rectangle
            surfaces = gmsh.model.getEntities(dim=2)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(dx/nx,dy/ny))
            # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(dx/nx,dy/ny))
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
            # Extrude each triangle into wedges
            for s in surfaces:
                gmsh.model.occ.extrude([s], 0, 0, dz, numElements=[1], recombine=True)

        else:
            # Step 2: Synchronize to access entities
            gmsh.model.occ.synchronize()

            # Step 3: Get boundary curves and assign transfinite curves
            boundary = gmsh.model.getBoundary([(2, rect)], oriented=False)
            for dim, tag in boundary:
                start = gmsh.model.getValue(dim, tag, [0.0])
                end = gmsh.model.getValue(dim, tag, [1.0])
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if abs(dx) > abs(dy):
                    gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)

            # Step 4: Transfinite + recombine surface
            gmsh.model.mesh.setTransfiniteSurface(rect)
            if element_type == 'hex':
                gmsh.model.mesh.setRecombine(2, rect)

            # Step 5: Extrude the surface to get a hexahedral block
            # Returns list of new volumes, surfaces, lines, points
            ext = gmsh.model.occ.extrude([(2, rect)], *[0, 0, dz], 
                                        numElements=[1], recombine=True)
            
    elif element_type == 'hex':
        # Step 2: Synchronize to access entities
        gmsh.model.occ.synchronize()

        # Step 3: Get boundary curves and assign transfinite curves
        boundary = gmsh.model.getBoundary([(2, rect)], oriented=False)
        for dim, tag in boundary:
            start = gmsh.model.getValue(dim, tag, [0.0])
            end = gmsh.model.getValue(dim, tag, [1.0])
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if abs(dx) > abs(dy):
                gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
            else:
                gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)

        # Step 4: Transfinite + recombine surface
        gmsh.model.mesh.setTransfiniteSurface(rect)
        
        gmsh.model.mesh.setRecombine(2, rect)

        # Step 5: Extrude the surface to get a hexahedral block
        # Returns list of new volumes, surfaces, lines, points
        ext = gmsh.model.occ.extrude([(2, rect)], *[0, 0, dz], 
                                    numElements=[1], recombine=True)
    elif element_type == 'tet':
        # Create box geometry of size n x n x 1
        lc = 1.0  # mesh size
        gmsh.model.occ.addBox(0, 0, 0, dx, dy, dz)
        gmsh.model.occ.synchronize()

        # Define transfinite meshing to get structured grid
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(dx/nx,dy/ny))
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", min(dx/nx,dy/ny))

        # Recombine OFF so mesh will be tetrahedral not hex
        # gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Generate and write mesh
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    
    if save is not None:

        if gmsh_version2:
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # Version 2.2 ASCII
        gmsh.write(f"{save}.msh")

    if display_mesh:
        if "-nopopup" not in sys.argv:
            gmsh.fltk.run()
    
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1,3) # Get Array of N,3 points
    elements = gmsh.model.mesh.getElementsByType(element_id[element_type])[1].reshape(-1,num_nodes_per_cell[element_type]) -1  # Get an array of (numcells, nodes per cell)
    gmsh.clear()
    gmsh.finalize()


    # Pyvista Stuff
    cells = np.hstack([np.full((elements.shape[0], 1), elements.shape[1],dtype = np.int32),elements],dtype = np.int32)
    pv_cell_type = pyvista_cell_id[element_type]

    return pv.UnstructuredGrid(cells,np.full(cells.shape[0],pv_cell_type,dtype = np.int32 ),nodes)

    # return nodes,elements,pv_cell_type



def define_boundary_walls(m:Mesh):
    faces = m.face_properties
    cells = m.cell_properties
    boundary_ids = m.face_properties.boundary_face_ids
    face_idx = faces.cell_face_index[boundary_ids,0]
    cell_ids = faces.adjacent_cells[boundary_ids,0]

    normals = cells.normal[cell_ids,face_idx]

    tol = 1e-3

    # Axis-aligned normal masks
    is_pos_x = np.abs(normals @ [1, 0, 0] - 1) < tol
    is_neg_x = np.abs(normals @ [-1, 0, 0] - 1) < tol
    is_pos_y = np.abs(normals @ [0, 1, 0] - 1) < tol
    is_neg_y = np.abs(normals @ [0, -1, 0] - 1) < tol
    is_pos_z = np.abs(normals @ [0, 0, 1] - 1) < tol
    is_neg_z = np.abs(normals @ [0, 0, -1] - 1) < tol

    # Step 5: Extract individual walls
    walls = {
        "+X": boundary_ids[is_pos_x],
        "-X": boundary_ids[is_neg_x],
        "+Y": boundary_ids[is_pos_y],
        "-Y": boundary_ids[is_neg_y],
        "+Z": boundary_ids[is_pos_z],
        "-Z": boundary_ids[is_neg_z],
    }
    
    m.add_groups(walls,'face')