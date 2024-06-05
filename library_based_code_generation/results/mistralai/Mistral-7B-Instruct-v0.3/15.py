 Here is a simplified Python code that follows your instructions. Note that you need to have Fealpy and SciPy installed. I've also assumed that you have defined the `CosSinData` function and the exact solutions for the time-harmonic equation.

```python
import argparse
import fealpy as fp
import numpy as np
from scipy.sparse.linalg import spsolve

def recover_curl(u):
    # Recover the curl of a given solution
    pass

def least_squares_matrix(x):
    # Calculate the least squares matrix for each node of a mesh
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--degree', type=int, default=2)
parser.add_argument('--h0', type=float, default=0.1)
parser.add_argument('--max_iters', type=int, default=5)
parser.add_argument('--theta', type=float, default=0.5)
args = parser.parse_args()

problem = fp.TimeHarmonicProblem2d(CosSinData)
mesh = fp.MeshFactory.create_box_mesh([0, 1], [0, 1], h=args.h0, quad_order=2)
mesh.remove_quadrant(3)

for _ in range(args.max_iters):
    V = fp.FirstKindNedelecFiniteElementSpace2d(mesh, degree=args.degree)
    u_h = fp.Function(V)
    bc = fp.DirichletBC(V, fp.Constant(0), fp.Boundary(0))
    fp.solve(problem, u_h, bc)

    u = u_h.vector()
    f = problem.f(u_h)
    A = problem.A(u_h, u_h)
    b = problem.b(u_h)

    error_l2 = np.linalg.norm(u - exact_solution)
    error_curl_exact = np.linalg.norm(recover_curl(exact_curl) - recover_curl(u_h))
    error_curl_recovered = np.linalg.norm(recover_curl(u_h) - recover_curl_recovered(u_h))

    if _ != args.max_iters - 1:
        markers = np.where(error_curl_recovered > args.theta * error_l2, 1, 0)
        mesh.refine(markers)

fp.plot(u_h, title='Error Rates')
fp.showmultirate(u_h, title='Error Rates')
```

This code is a starting point and requires you to fill in the `recover_curl` and `recover_curl_recovered` functions, as well as defining the exact solutions for the time-harmonic equation. Additionally, you may need to adjust the code to fit your specific problem and requirements.