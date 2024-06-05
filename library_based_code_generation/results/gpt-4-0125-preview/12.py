```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh.mesh_tools import find_entity_barycenter, find_entity_measure

# Define nodes and cells for the TetrahedronMesh
nodes = np.array([
    [0, 0, 0],  # Node 0
    [1, 0, 0],  # Node 1
    [0, 1, 0],  # Node 2
    [0, 0, 1]   # Node 3
], dtype=np.float)

cells = np.array([
    [0, 1, 2, 3]  # Cell connecting nodes 0, 1, 2, and 3
], dtype=np.int)

# Create the TetrahedronMesh
mesh = TetrahedronMesh(nodes, cells)

# Calculate number of nodes, edges, faces, and cells
num_nodes = mesh.number_of_nodes()
num_edges = mesh.number_of_edges()
num_faces = mesh.number_of_faces()
num_cells = mesh.number_of_cells()

# Calculate and store barycenter coordinates for each entity
node_barycenter = find_entity_barycenter(mesh, entity_dim=0)
edge_barycenter = find_entity_barycenter(mesh, entity_dim=1)
face_barycenter = find_entity_barycenter(mesh, entity_dim=2)
cell_barycenter = find_entity_barycenter(mesh, entity_dim=3)

# Calculate and store the measure of each entity
node_measure = find_entity_measure(mesh, entity_dim=0)
edge_measure = find_entity_measure(mesh, entity_dim=1)
face_measure = find_entity_measure(mesh, entity_dim=2)
cell_measure = find_entity_measure(mesh, entity_dim=3)

# Relationships between entities
cell_to_cell = mesh.ds.cell_to_cell()
cell_to_face = mesh.ds.cell_to_face()
cell_to_edge = mesh.ds.cell_to_edge()
cell_to_node = mesh.ds.cell_to_node()

# Identify boundary flags
is_boundary_node = mesh.ds.boundary_node_flag()
is_boundary_edge = mesh.ds.boundary_edge_flag()
is_boundary_face = mesh.ds.boundary_face_flag()
is_boundary_cell = mesh.ds.boundary_cell_flag()

# Indices of boundary nodes, edges, faces, and cells
boundary_nodes = np.where(is_boundary_node)[0]
boundary_edges = np.where(is_boundary_edge)[0]
boundary_faces = np.where(is_boundary_face)[0]
boundary_cells = np.where(is_boundary_cell)[0]

# Plot the mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh.add_plot(ax)
mesh.find_node(ax, showindex=True)
mesh.find_edge(ax, showindex=True)
mesh.find_cell(ax, showindex=True)
plt.show()
```