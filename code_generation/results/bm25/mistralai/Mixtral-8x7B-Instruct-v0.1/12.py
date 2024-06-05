 import fealpy
import matplotlib.pyplot as plt

mesh = fealpy.TetrahedronMesh("../mesh/tet.msh")

nodes = mesh.nodes
cells = mesh.cells
cell2node = mesh.cell2node

n_nodes = mesh.number_of_nodes
n_edges = mesh.number_of_edges
n_faces = mesh.number_of_faces
n_cells = mesh.number_of_cells

n_nodes_of_edges = mesh.number_of_nodes_of_edges
n_nodes_of_faces = mesh.number_of_nodes_of_faces
n_nodes_of_cells = mesh.number_of_nodes_of_cells
n_edges_of_faces = mesh.number_of_edges_of_faces
n_edges_of_cells = mesh.number_of_edges_of_cells
n_faces_of_cells = mesh.number_of_faces_of_cells

n_boundary_nodes = mesh.number_of_boundary_nodes
n_boundary_edges = mesh.number_of_boundary_edges

boundary_flags = mesh.boundary_flags

boundary_nodes_indices = mesh.boundary_nodes_indices
boundary_edges_indices = mesh.boundary_edges_indices

quad_pts, quad_wts = mesh.get_quadrature_points_and_weights(3)
gauss_pt, gauss_wt = mesh.get_gauss_point_and_weight(1)

barycenter_nodes = nodes[cell2node] @ gauss_wt
barycenter_edges = nodes[boundary_edges_indices] @ quad_wts[n_nodes_of_edges:]
barycenter_faces = nodes[boundary_faces_indices] @ quad_wts[n_nodes_of_faces:]
barycenter_cells = nodes[cells] @ quad_wts[n_nodes_of_cells:]

measure_nodes = fealpy.volume(nodes[cell2node], cells) @ gauss_wt
measure_edges = fealpy.area(nodes[boundary_edges_indices], boundary_edges_indices) @ quad_wts[n_nodes_of_edges:]
measure_faces = fealpy.area(nodes[boundary_faces_indices], boundary_faces_indices) @ quad_wts[n_nodes_of_faces:]
measure_cells = fealpy.volume(nodes[cells], cells) @ quad_wts[n_nodes_of_cells:]

cell2cell = fealpy.connect_cell_to_cell(cells)
cell2face = fealpy.connect_cell_to_face(cells, faces)
face2cell = fealpy.connect_face_to_cell(faces, cells)
cell2edge = fealpy.connect_cell_to_edge(cells, edges)
edge2cell = fealpy.connect_edge_to_cell(edges, cells)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'ko')
for i in range(n_cells):
 axes.text(nodes[cell2node[i, 0], 0], nodes[cell2node[i, 0], 1], nodes[cell2node[i, 0], 2], str(i), color='r')
for i in range(0, n_cells*4, 4):
 axes.plot(nodes[cell2node[cells[i//4], :], 0], nodes[cell2node[cells[i//4], :], 1], nodes[cell2node[cells[i//4], :], 2], 'k-')
plt.show()