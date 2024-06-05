```python
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, showmesh
from fealpy.model.heat_model_2d import HeatConductionModel2d

# User-defined parameters
num_spatial_divisions = int(input("Enter the number of spatial divisions: "))
num_temporal_divisions = int(input("Enter the number of temporal divisions: "))
adaptive_refinement_stop_threshold = float(input("Enter the adaptive refinement stop threshold: "))
adaptive_refinement_parameter = float(input("Enter the adaptive refinement parameter: "))
adaptive_coarsening_parameter = float(input("Enter the adaptive coarsening parameter: "))

# Model and mesh
model = HeatConductionModel2d()
mesh = MF.boxmesh2d(model.domain(), nx=num_spatial_divisions, ny=num_spatial_divisions, meshtype='tri')

# Time discretization
timeline = UniformTimeLine(0, 1, num_temporal_divisions)

# Finite Element space
space = LagrangeFiniteElementSpace(mesh, p=1)

# Dirichlet boundary condition
bc = DirichletBC(space, model.dirichlet)

# Time stepping
for i in range(num_temporal_divisions):
    t1 = timeline.next_time_level()
    print(f"Time step {i+1}/{num_temporal_divisions}, Time: {t1}")
    
    # Solve the linear system
    A = space.stiff_matrix(c=model.c)
    b = np.zeros(space.number_of_global_dofs())
    bc.apply(A, b, t1)
    x = np.linalg.solve(A, b)
    
    # Compute error and adaptively refine and coarsen the mesh
    error = np.max(np.abs(x - model.solution(mesh.node)))
    print(f"Error at time step {i+1}: {error}")
    
    if error > adaptive_refinement_stop_threshold:
        is_refined = space.adaptive_refine(adaptive_refinement_parameter)
        if is_refined:
            print("Mesh refined")
            showmesh(mesh)
            plt.savefig(f"mesh_refinement_step_{i+1}.png")
            plt.close()
    
    is_coarsened = space.adaptive_coarsen(adaptive_coarsening_parameter)
    if is_coarsened:
        print("Mesh coarsened")
        showmesh(mesh)
        plt.savefig(f"mesh_coarsening_step_{i+1}.png")
        plt.close()
    
    # Plot solution at specified time steps
    if (i+1) % (num_temporal_divisions // 4) == 0 or i == num_temporal_divisions - 1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        space.function_plot(ax, x)
        plt.title(f"Numerical solution at time {t1}")
        plt.savefig(f"numerical_solution_at_time_{t1}.png")
        plt.close()
```