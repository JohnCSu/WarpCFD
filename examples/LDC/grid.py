import numpy as np
import warp as wp
from warp_cfd.preprocess.mesh import Mesh
import pyvista as pv
from pyvista import CellType

def _make_hex_grid(nx, ny, nz, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0)):
    '''Vibe coded so not at all effecient'''
    # Step 1: Create a structured grid (VTK_VOXEL)
    img = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=spacing, origin=origin)
    ugrid = img.cast_to_unstructured_grid()

    # Step 2: Convert voxel cells to hexahedron cells
    def voxel_to_hex(ugrid):
        new_cells = []
        new_types = []
        reorder = [0, 1, 3, 2, 4, 5, 7, 6]  # Voxel to Hex point ordering
        for i in range(ugrid.n_cells):
            cell = ugrid.get_cell(i)
            pts = cell.point_ids
            new_cells.append([8] + [pts[j] for j in reorder])
            new_types.append(CellType.HEXAHEDRON)
        cell_array = np.hstack(new_cells)
        return pv.UnstructuredGrid(cell_array, new_types, ugrid.points)

    hex_grid = voxel_to_hex(ugrid)
    return hex_grid


def create_hex_grid(nx, ny, nz, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0)):
    '''Use to create a grid of cube cells'''
    # Create a 4×4×2 hex grid
    hex_grid = _make_hex_grid(nx, ny, nz, spacing, origin)
    # Optional: add cell IDs for visualization
    hex_grid.cell_data["id"] = np.arange(hex_grid.n_cells)
    
    m = Mesh(hex_grid)
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
    
    return m


if __name__ == '__main__':
    
    m = create_hex_grid(10,10,1,(0.1,0.1,0.1))
    get_vtk_faces = lambda boundary_ids,faces_array : np.hstack(np.concatenate((np.ones(( boundary_ids.shape[0],1),dtype=np.int32)*4,faces_array),axis = 1))


    faces = m.face_properties
    boundary_ids = faces.boundary_face_ids

    nodes_list = np.unique(faces.unique_faces[boundary_ids])
    points = m.nodes[nodes_list]


    verts = pv.PolyData(points)

    plotter = pv.Plotter()
    plotter.add_mesh(m.pyvista_mesh, color="lightgray", show_edges=True)

    colors = ["red", "blue", "green", "yellow", "purple", "cyan"]
    for i, (name, wall) in enumerate(m.groups.items()):
        wall = wall.ids
        vtk_faces = get_vtk_faces(wall,faces.unique_faces[wall])
        face_polydata = pv.PolyData(points,faces = vtk_faces)
        plotter.add_mesh(face_polydata, color=colors[i], label=name, show_edges=True)

    # plotter.add_mesh(face_polydata, color="red", label="Z")
    plotter.add_mesh(verts,color = 'black')
    plotter.add_legend()
    plotter.show()