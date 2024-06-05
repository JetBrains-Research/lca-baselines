import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh

node = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

cell = np.array([
    [0, 1, 2, 3],
    [1, 4, 5, 3],
    [1, 5, 6, 4],
    [3, 5, 6, 7]
])

tmesh = TetrahedronMesh(node, cell)

num_nodes = tmesh.number_of_nodes()
num_edges = tmesh.number_of_edges()
num_faces = tmesh.number_of_faces()
num_cells = tmesh.number_of_cells()

barycenter_nodes = tmesh.entity_barycenter(0)
barycenter_edges = tmesh.entity_barycenter(1)
barycenter_faces = tmesh.entity_barycenter(2)
barycenter_cells = tmesh.entity_barycenter(3)

measure_nodes = tmesh.entity_measure(0)
measure_edges = tmesh.entity_measure(1)
measure_faces = tmesh.entity_measure(2)
measure_cells = tmesh.entity_measure(3)

cell_to_cell = tmesh.ds.cell_to_cell()
cell_to_face = tmesh.ds.cell_to_face()
face_to_cell = tmesh.ds.face_to_cell()

boundary_flags = tmesh.ds.boundary_flag()
boundary_nodes = tmesh.ds.boundary_node_index()
boundary_edges = tmesh.ds.boundary_edge_index()
boundary_faces = tmesh.ds.boundary_face_index()
boundary_cells = tmesh.ds.boundary_cell_index()

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
tmesh.add_plot(axes)
tmesh.find_point(axes, showindex=True)
plt.show()