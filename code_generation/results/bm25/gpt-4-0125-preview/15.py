```python
import numpy as np
import argparse
from fealpy.mesh import MeshFactory
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate
from scipy.sparse.linalg import spsolve
from fealpy.timeharmonicproblem import CosSinData

def parse_arguments():
    parser = argparse.ArgumentParser(description="Solve the time-harmonic equation using adaptive methods.")
    parser.add_argument("--degree", type=int, default=1, help="Degree of the Nedelec element.")
    parser.add_argument("--mesh_size", type=float, default=0.1, help="Initial mesh size.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of adaptive iterations.")
    parser.add_argument("--theta", type=float, default=0.3, help="Theta parameter for adaptive iteration.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    pde = CosSinData()
    mf = MeshFactory()
    mesh = mf.boxmesh2d([0, 1, 0, 1], nx=int(1/args.mesh_size), ny=int(1/args.mesh_size), meshtype='tri')
    mesh.remove_cells(mesh.entity('cell', index=np.where(mesh.entity_barycenter('cell')[:, 0] + mesh.entity_barycenter('cell')[:, 1] > 1)[0]))
    
    for i in range(args.max_iter):
        space = FirstKindNedelecFiniteElementSpace2d(mesh, p=args.degree)
        bc = DirichletBC(space, pde.dirichlet)
        
        A = space.stiff_matrix()
        F = space.source_vector(pde.source)
        
        A, F = bc.apply(A, F)
        uh = spsolve(A, F)
        
        errorMatrix = space.error_matrix(pde.solution, uh)
        L2Error = np.sqrt(np.dot(uh, errorMatrix@uh))
        
        print(f"Iteration {i+1}, L2 Error: {L2Error}")
        
        if i < args.max_iter - 1:
            # Mark and refine mesh based on recovery error
            pass  # Placeholder for adaptive mesh refinement logic
    
    # Plot error rates
    # showmultirate(...)  # Placeholder for plotting logic

if __name__ == "__main__":
    main()
```