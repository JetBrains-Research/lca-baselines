 import fealpy
import matplotlib.pyplot as plt
import numpy as np

mesh = fealpy.TetrahedronMesh('unit_cube.msh')

nodes = mesh.nodes
cells = mesh.elements
n_nodes = nodes.shape[0]
n_cells = cells.shape[0]

edges = fealpy.get_edges(cells)
faces = fealpy.get_faces(cells)
n_edges = edges.shape[0]
n_faces = faces.shape[0]

node_measure = fealpy.measure_node(nodes)
cell_measure = fealpy.measure_cell(nodes, cells)
edge_measure = fealpy.measure_edge(nodes, edges)
face_measure = fealpy.measure_face(nodes, faces)

entity_barycenter = {
    'node': nodes,
    'cell': fealpy.barycenter_cell(nodes, cells),
    'edge': fealpy.barycenter_edge(nodes, edges),
    'face': fealpy.barycenter_face(nodes, faces)
}

entity_relations = {
    'cell_to_cell': fealpy.get_cell_to_cell(cells),
    'cell_to_face': fealpy.get_cell_to_face(cells, faces),
    'cell_to_edge': fealpy.get_cell_to_edge(cells, edges),
    'face_to_cell': fealpy.get_face_to_cell(faces, cells),
    'face_to_face': fealpy.get_face_to_face(faces),
    'face_to_edge': fealpy.get_face_to_edge(faces, edges),
    'edge_to_cell': fealpy.get_edge_to_cell(edges, cells),
    'edge_to_edge': fealpy.get_edge_to_edge(edges)
}

entity_flags = {
    'node': np.zeros(n_nodes, dtype=int),
    'cell': np.zeros(n_cells, dtype=int),
    'edge': np.zeros(n_edges, dtype=int),
    'face': np.zeros(n_faces, dtype=int)
}

boundary_nodes, boundary_flags = fealpy.get_boundary_nodes(nodes, cells, entity_flags['node'])
boundary_edges, boundary_flags = fealpy.get_boundary_edges(nodes, edges, entity_flags['edge'])
boundary_faces, boundary_flags = fealpy.get_boundary_faces(nodes, faces, entity_flags['face'])
boundary_cells, boundary_flags = fealpy.get_boundary_cells(nodes, cells, entity_flags['cell'])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'ko')
for cell in cells:
    ax.plot(nodes[cell, 0], nodes[cell, 1], nodes[cell, 2], 'b-')
plt.show()

print("Number of nodes:", n_nodes)
print("Number of cells:", n_cells)
print("Number of edges:", n_edges)
print("Number of faces:", n_faces)
print("Boundary nodes:", boundary_nodes.shape[0])
print("Boundary edges:", boundary_edges.shape[0])
print("Boundary faces:", boundary_faces.shape[0])
print("Boundary cells:", boundary_cells.shape[0])