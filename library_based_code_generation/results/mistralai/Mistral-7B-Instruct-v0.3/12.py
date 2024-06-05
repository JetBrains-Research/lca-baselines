 ```python
import fealpy as fp
import numpy as np

# Create TetrahedronMesh
mesh = fp.Mesh.TetrahedronMesh(np.zeros((4, 3)), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))

# Calculate number of nodes, edges, faces, and cells
num_nodes = mesh.num_vertices()
num_edges = mesh.num_edges()
num_faces = mesh.num_faces()
num_cells = mesh.num_cells()

# Calculate barycenter coordinates for each entity
barycenters = mesh.barycenters()

# Calculate measure of each entity
volumes = mesh.cell_volumes()
areas = mesh.face_areas()
lengths = mesh.edge_lengths()

# Store relationships between each entity
cell_faces = mesh.cell_faces()
cell_cells = mesh.cell_cells()

# Identify boundary flags for each entity
boundary_flags = mesh.boundary_flags()

# Identify indices of boundary nodes, edges, faces, and cells
boundary_nodes = mesh.boundary_nodes()
boundary_edges = mesh.boundary_edges()
boundary_faces = mesh.boundary_faces()
boundary_cells = mesh.boundary_cells()

# Plot the mesh using matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh.plot(ax=ax)
ax.scatter(barycenters[:, 0], barycenters[:, 1], barycenters[:, 2], c='r', s=100)
for i in range(num_nodes):
    ax.text(barycenters[i, 0], barycenters[i, 1], barycenters[i, 2], str(i), fontdict={'size': 10})
plt.show()
```