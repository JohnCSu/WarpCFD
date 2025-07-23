from warp_cfd.preprocess.grid import create_2D_grid
from warp_cfd.preprocess.mesh import Mesh
import pyvista as pv
import numpy as np

pv_mesh = create_2D_grid((0,0,0),1,1,1,1,element_type= 'wedge',unstructured_wedge = False,display_mesh=False)

m = Mesh(pv_mesh)

int_dtype = np.int32
wedge_faces = np.array([
                [0, 1, 2,-1],
                [5, 4, 3,-1],
                [0, 2, 5,3],
                [1,4,5,2],
                [0, 3, 4,1],
            ],dtype=int_dtype)

face_ids = m.cell_properties.faces
normals = m.cell_properties.normal
centroids = m.face_properties.centroid

C, F = face_ids.shape
face_centroids = np.full((C, F, 3), np.nan)  # Or any default

# Valid mask
valid = face_ids != -1

# Indexing only valid positions
rows, cols = np.nonzero(valid)
face_centroids[rows, cols] = centroids[face_ids[rows, cols]]

centroids_flat = face_centroids[0].reshape(-1, 3)
normals_flat = normals[0].reshape(-1, 3)
valid = ~np.isnan(normals_flat).any(axis=1)

points = pv.PolyData(centroids_flat)
points["normals"] = normals_flat
arrows = points.glyph(orient="normals", scale=False, factor=0.2)  # set `factor` as desired
# Plot
print(m.cells)
print(centroids_flat)
print(normals_flat)
node_labels = [f'node {i}' for i in range(len(m.nodes))]
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, show_edges=True, color='lightgray',opacity = 0.3)
plotter.add_mesh(arrows, color='red')
plotter.add_point_labels(centroids, list(range(len(centroids))), font_size=10, point_size=10, text_color='black')
plotter.add_point_labels(m.nodes,node_labels , font_size=10, point_size=10, text_color='black')
plotter.show()