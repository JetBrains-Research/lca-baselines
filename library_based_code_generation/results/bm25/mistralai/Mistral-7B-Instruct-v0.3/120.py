 Here is a simplified Python code using Fealpy library for the given problem:

```python
import argparse
import fealpy as fp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--theta', type=float, default=0.5)
parser.add_argument('--k', type=int, default=3)
args = parser.parse_args()

max_iter = args.max_iter
theta = args.theta
k = args.k

# Define the friction problem
mesh = fp.Mesh.create_halfedge_polygon_mesh(n=100, theta=theta)
V, F, E = mesh.get_vertex_list(), mesh.get_face_list(), mesh.get_edge_list()

space = fp.FunctionSpace(mesh, 'P', k)
u = space.function('u')

A, b = fp.assemble_for_sspace_and_vspace_with_vector_basis(space, space, u, u, test_dirichlet_and_neumann_bc_on_halfedge_polygon_mesh)

# Loop over iterations
for i in range(max_iter):
    fp.solve(A, b, u)

    # Calculate residuals and high order terms
    r = fp.assemble_for_sspace_and_vspace_with_vector_basis(space, space, u, 1 - u, test_dirichlet_and_neumann_bc_on_halfedge_polygon_mesh)
    h = fp.assemble_for_sspace_and_vspace_with_vector_basis(space, space, u, u**2, test_dirichlet_and_neumann_bc_on_halfedge_polygon_mesh)

    # Save results and error data to a file
    data = {'u': u.vector(), 'r': r, 'h': h}
    fp.save_data(data, f'results/iter_{i}.dat')

    # Refine the mesh based on the residuals
    if i < max_iter - 1:
        refined_mesh = fp.ATriMesher(mesh).refine(r)
        mesh = refined_mesh
        V, F, E = refined_mesh.get_vertex_list(), refined_mesh.get_face_list(), refined_mesh.get_edge_list()
        space = fp.FunctionSpace(mesh, 'P', k)
        u = space.function('u')
        A, b = fp.assemble_for_sspace_and_vspace_with_vector_basis(space, space, u, u, test_dirichlet_and_neumann_bc_on_halfedge_polygon_mesh)

# Save final error data to a file
data = {'u': u.vector(), 'r': r, 'h': h}
fp.save_data(data, 'results/final.dat')

# Plot the mesh and save it as an image file
fp.plot(mesh, to_vtk_file='mesh.vtk')

# Display a multi-rate plot
fp.plot_multi_rate(data, 'results/multi_rate.png')
```

This code assumes that you have already defined the `test_dirichlet_and_neumann_bc_on_halfedge_polygon_mesh` function for setting the boundary conditions on the halfedge polygon mesh. You may need to adjust the code according to your specific problem setup.