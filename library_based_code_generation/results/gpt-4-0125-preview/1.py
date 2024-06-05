```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.tools.show import showmultirate, showmesh
from fealpy.pde.heat_conduction_model_2d import HeatConductionModel2d

# User-defined parameters
num_spatial_divisions = int(input("Enter the number of spatial divisions: "))
num_temporal_divisions = int(input("Enter the number of temporal divisions: "))
adaptive_refinement_stop_threshold = float(input("Enter the adaptive refinement stop threshold: "))
adaptive_refinement_parameter = float(input("Enter the adaptive refinement parameter: "))
adaptive_coarsening_parameter = float(input("Enter the adaptive coarsening parameter: "))

# Model and mesh
pde = HeatConductionModel2d()
mesh = pde.init_mesh(n=num_spatial_divisions)
timeline = UniformTimeLine(0, 1, num_temporal_divisions)

# Finite Element Space
space = LagrangeFiniteElementSpace(mesh, p=1)

# Dirichlet boundary condition
bc = DirichletBC(space, pde.dirichlet)

# Time stepping
for i in range(num_temporal_divisions):
    t1 = timeline.next_time_level()
    print(f"Time step {i+1}/{num_temporal_divisions}, Time: {t1}")
    
    # Solve the PDE
    uh = space.function()
    A = space.stiff_matrix(c=pde.c)
    F = space.source_vector(pde.source)
    bc.apply(A, F, uh)
    uh[:] = np.linalg.solve(A, F)
    
    # Error and adaptive refinement
    error = np.max(np.abs(uh - pde.solution(mesh.node)))
    print(f"Error: {error}")
    
    if error > adaptive_refinement_stop_threshold:
        # Adaptive refinement
        isMarkedCell = space.recovery_estimate(uh, eta=adaptive_refinement_parameter)
        mesh.refine_triangle_rg(isMarkedCell)
        space = LagrangeFiniteElementSpace(mesh, p=1)
    else:
        # Adaptive coarsening
        isMarkedCell = space.recovery_estimate(uh, eta=adaptive_coarsening_parameter)
        mesh.coarsen_triangle_rg(isMarkedCell)
        space = LagrangeFiniteElementSpace(mesh, p=1)
    
    # Plotting
    plt.figure()
    showmesh(mesh)
    plt.title(f"Mesh at time step {i+1}")
    plt.savefig(f"mesh_at_step_{i+1}.png")
    plt.close()

    if (i+1) % (num_temporal_divisions // 5) == 0 or i == num_temporal_divisions - 1:
        plt.figure()
        space.function(uh).plot()
        plt.title(f"Numerical solution at time step {i+1}")
        plt.savefig(f"solution_at_step_{i+1}.png")
        plt.close()
```