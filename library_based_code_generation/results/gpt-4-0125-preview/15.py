```python
import numpy as np
import argparse
from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
from scipy.sparse.linalg import spsolve
from fealpy.model import CosSinData

def parse_arguments():
    parser = argparse.ArgumentParser(description="Solve the time-harmonic equation using adaptive methods.")
    parser.add_argument("--degree", type=int, default=1, help="Degree of the Nedelec element.")
    parser.add_argument("--mesh_size", type=float, default=0.1, help="Initial mesh size.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of adaptive iterations.")
    parser.add_argument("--theta", type=float, default=0.3, help="Theta parameter for adaptive iteration.")
    return parser.parse_args()

def recover_curl(solution):
    # Placeholder for curl recovery function
    pass

def calculate_least_squares_matrix(node):
    # Placeholder for least squares matrix calculation
    pass

def main():
    args = parse_arguments()
    
    model = CosSinData()
    mesh = MeshFactory.boxmesh2d([0, 1, 0, 1], nx=args.mesh_size, ny=args.mesh_size, meshtype='tri')
    mesh.remove_cells(mesh.entity_barycenter('cell')[:, 0] + mesh.entity_barycenter('cell')[:, 1] > 1)
    
    error_rates = []
    for i in range(args.max_iter):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=args.degree)
        bc = DirichletBC(space, model.dirichlet)
        
        A = space.stiff_matrix(c=model.c)
        F = space.source_vector(model.source)
        
        A, F = bc.apply(A, F)
        solution = spsolve(A, F)
        
        # Compute errors and recovery
        # Placeholder for error computation and mesh refinement
        
        if i < args.max_iter - 1:
            # Placeholder for marking cells and refining the mesh
            pass
    
    # Placeholder for plotting error rates
    showmultirate(plt, 0, error_rates)

if __name__ == "__main__":
    main()
```