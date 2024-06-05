 Here is the Python code that fulfills your requirements using the Fealpy and Matplotlib libraries:

```python
import fealpy as fp
import matplotlib.pyplot as plt

# Create a TetrahedronMesh
mesh = fp.Mesh.TetrahedronMesh(np.zeros((4, 3)), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))

# Define nodes, cells, and their properties
nodes = mesh.get_nodes()
cells = mesh.get_cells()

# Calculate the number of nodes, edges, faces, and cells
num_nodes = mesh.number_of_nodes()
num_edges = mesh.number_of_edges()
num_faces = mesh.number_of_faces()
num_cells = mesh.number_of_cells()

# Calculate barycenter coordinates for each entity
barycenters = {}
for cell in cells:
    barycenters[cell] = fp.function.Constant(mesh, fp.assemble(fp.function.Constant(mesh, 1.) * fp.function.CellAffine(cell)))

# Calculate the measure of each entity
measures = {}
for cell in cells:
    measures[cell] = fp.measure(cell)

# Store relationships between each entity
cell_to_cell = {}
cell_to_face = {}
for cell in cells:
    cell_to_cell[cell] = []
    cell_to_face[cell] = []
    for face in cell.faces:
        cell_to_cell[cell].append(face.cell)
        cell_to_face[cell].append(face)

# Identify boundary flags for each entity
boundary_nodes = mesh.get_boundary_nodes()
boundary_edges = mesh.get_boundary_edges()
boundary_faces = [face for cell in mesh.get_boundary_cells() for face in cell.faces]
boundary_cells = mesh.get_boundary_cells()

# Plot the mesh using matplotlib
fig, ax = plt.subplots()
for cell in cells:
    if cell in boundary_cells:
        fp.plot.plot_cell(ax, cell, node_labels=cell.vertices, edge_labels=cell.edges, cell_label=cell)
    else:
        fp.plot.plot_cell(ax, cell, node_labels=cell.vertices, edge_labels=cell.edges)
plt.show()
```

This code creates a TetrahedronMesh, calculates the number of nodes, edges, faces, and cells, and stores their barycenter coordinates, measures, and relationships. It also identifies boundary flags for each entity and the indices of boundary nodes, edges, faces, and cells. Finally, it plots the mesh using Matplotlib, showing the indices of nodes, edges, and cells.