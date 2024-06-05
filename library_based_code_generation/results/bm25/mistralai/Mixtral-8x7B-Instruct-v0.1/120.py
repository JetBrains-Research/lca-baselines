 import sys
import fealpy
import matplotlib.pyplot as plt
import numpy as np
import os

max_iter = int(sys.argv[1])
theta = float(sys.argv[2])
k = int(sys.argv[3])

problem = fealpy.SimpleFrictionProblem(2)
mesh = fealpy.UniformRefinedMesh(problem.domain, 1)

fealpy.test.test_dirichlet_and_neumann_bc_on_interval_mesh(problem, mesh)

data_file = os.path.join(fealpy.test_dir, 'simplified_friction_%d_%f_%d.dat' % (theta, k, max_iter))
error_file = os.path.join(fealpy.test_dir, 'simplified_friction_%d_%f_%d_error.dat' % (theta, k, max_iter))

with open(data_file, 'w') as f:
    f.write('Iteration, Residual, H1_error, L2_error\n')

for iter in range(max_iter):
    uh = fealpy.functionspace.LagrangeFiniteElementSpace(mesh, k)
    bd = fealpy.functionspace.BoundaryFunctionSpace(uh)
    u, v = uh.TrialFunction(), uh.TestFunction()
    a = fealpy.formspace.BilinearForm(uh, bd)
    a_u = fealpy.formspace.LinearForm(uh, bd)
    a_u += problem.assemble_cell_righthand_side(uh, mesh)
    a += problem.assemble_cell_dof_matrix(uh, mesh)
    a += problem.assemble_boundary_term(uh, bd, mesh)
    a.finalize()
    a_u.finalize()

    uh.zero()
    a_u.apply(uh)

    if iter > 0:
        residual = np.linalg.norm(a_u.vec)
        H1_error = fealpy.error.H1_error(problem.exact_solution, uh, mesh)
        L2_error = fealpy.error.L2_error(problem.exact_solution, uh, mesh)

        with open(data_file, 'a') as f:
            f.write('%d, %.16e, %.16e, %.16e\n' % (iter, residual, H1_error, L2_error))

        print('Iteration %d: Residual = %.16e, H1_error = %.16e, L2_error = %.16e' % (iter, residual, H1_error, L2_error))

    if iter < max_iter - 1:
        mesh.uniform_refine()

uh.plot(colorbar=True)
plt.savefig(os.path.join(fealpy.test_dir, 'simplified_friction_%d_%f_%d_mesh.png' % (theta, k, max_iter)))
plt.close()

uh.save_to_file(os.path.join(fealpy.test_dir, 'simplified_friction_%d_%f_%d_sol.vtu' % (theta, k, max_iter)))

with open(error_file, 'w') as f:
    f.write('Iteration, H1_error, L2_error\n')
    for line in open(data_file):
        if line.startswith(b'Iteration,'):
            continue
        fields = line.split(',')
        f.write('%s, %.16e, %.16e\n' % (fields[0], fields[2], fields[3]))

fealpy.plot.plot_data(error_file, title='Simplified Friction Problem Error History', xlabel='Iteration', ylabel='Error')