```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh

# Define nodes and cells for the TetrahedronMesh
nodes = np.array([
    [0, 0, 0],  # Node 0
    [1, 0, 0],  # Node 1
    [0, 1, 0],  # Node 2
    [0, 0, 1]   # Node 3
])

cells = np.array([
    [0, 1, 2, 3]  # Cell connecting nodes 0, 1, 2, 3
])

# Create the TetrahedronMesh
mesh = TetrahedronMesh(nodes, cells)

# Calculate and store the number of nodes, edges, faces, and cells
num_nodes = mesh.number_of_nodes()
num_edges = mesh.number_of_edges()
num_faces = mesh.number_of_faces()
num_cells = mesh.number_of_cells()

# Calculate and store the barycenter coordinates for each entity
node_barycenter = mesh.entity_barycenter('node')
edge_barycenter = mesh.entity_barycenter('edge')
face_barycenter = mesh.entity_barycenter('face')
cell_barycenter = mesh.entity_barycenter('cell')

# Calculate and store the measure of each entity
node_measure = mesh.entity_measure('node')
edge_measure = mesh.entity_measure('edge')
face_measure = mesh.entity_measure('face')
cell_measure = mesh.entity_measure('cell')

# Relationships between entities
cell_to_cell = mesh.ds.cell_to_cell()
cell_to_face = mesh.ds.cell_to_face()
cell_to_edge = mesh.ds.cell_to_edge()
cell_to_node = mesh.ds.cell_to_node()

# Boundary flags
boundary_nodes = mesh.ds.boundary_nodes()
boundary_edges = mesh.ds.boundary_edges()
boundary_faces = mesh.ds.boundary_faces()
boundary_cells = mesh.ds.boundary_cells()

# Plotting the mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2])  # Plot nodes

# Annotate nodes
for i, txt in enumerate(range(num_nodes)):
    ax.text(nodes[i, 0], nodes[i, 1], nodes[i, 2], '%d' % txt, size=20, zorder=1)

# Annotate cells (using cell barycenters)
for i, txt in enumerate(range(num_cells)):
    ax.text(cell_barycenter[i, 0], cell_barycenter[i, 1], cell_barycenter[i, 2], 'C%d' % txt, color='red')

plt.show()
```