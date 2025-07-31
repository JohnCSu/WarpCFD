from warp_cfd.preprocess.grid import create_2D_grid
from warp_cfd.preprocess.mesh import Mesh
import pyvista as pv
import numpy as np
n = 3
pv_mesh = create_2D_grid((0,0,0),n,n,1,1,element_type= 'hex',unstructured_wedge = False,display_mesh=False)
show_cell_ids = False
show_nodes = False
show_face_normals = True
m = Mesh(pv_mesh)
face_ids = m.cell_properties.faces
normals = m.cell_properties.normal
centroids = m.face_properties.centroid

C, F = face_ids.shape
face_centroids = np.full((C, F, 3), np.nan)  # Or any default


node_labels = [f'node {i}' for i in range(len(m.nodes))]
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, show_edges=True, color='lightgray',opacity = 0.3)
plotter.add_axes_at_origin()

# Show Cell ID
if show_cell_ids:
    plotter.add_point_labels(m.cell_centroids,list(range(len(m.cell_centroids))) , font_size=10, point_size=10, text_color='black')

if show_face_normals:
    # Shows faces and normals
    # Valid mask
    valid = face_ids != -1

    # Indexing only valid positions
    rows, cols = np.nonzero(valid)
    face_centroids[rows, cols] = centroids[face_ids[rows, cols]]

    k = 0
    centroids_flat = face_centroids[k].reshape(-1, 3)
    normals_flat = normals[k].reshape(-1, 3)
    valid = ~np.isnan(normals_flat).any(axis=1)

    points = pv.PolyData(centroids_flat)
    points["normals"] = normals_flat
    arrows = points.glyph(orient="normals", scale=False, factor=0.2)  # set `factor` as desired
    # Plot
    print(m.cells)
    print(centroids_flat)
    print(normals_flat)

    plotter.add_mesh(arrows, color='red')
    plotter.add_point_labels(centroids, list(range(len(centroids))), font_size=10, point_size=10, text_color='black')

if show_nodes: 
    plotter.add_point_labels(m.nodes,node_labels , font_size=10, point_size=10, text_color='black')

plotter.show()