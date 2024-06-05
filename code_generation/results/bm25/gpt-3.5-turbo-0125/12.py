import numpy as np
import matplotlib.pyplot as plt
import fealpy as fl

node = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2]
])

cell = np.array([
    [0, 1, 2]
])

tmesh = fl.TriangleMesh(node, cell)

print("Number of nodes:", tmesh.number_of_nodes())
print("Number of edges:", tmesh.number_of_edges())
print("Number of faces:", tmesh.number_of_faces())
print("Number of cells:", tmesh.number_of_cells())

tmesh.init_level_set_function()
tmesh.find_node()
tmesh.find_edge()
tmesh.find_face()
tmesh.find_cell()

tmesh.init_level_set_function()
tmesh.find_node()
tmesh.find_edge()
tmesh.find_face()
tmesh.find_cell()

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
tmesh.find_node()
tmesh.find_edge()
tmesh.find_face()
tmesh.find_cell()
plt.show()