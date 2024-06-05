 Here is the Python code that follows your instructions, using the TACS library:

```python
import numpy as np
import os
from mpi4py import MPI
from tacs import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the mesh from a BDF file
mesh = MeshLoader('mesh.bdf')

# Set constitutive properties
density = 7850.0
E = 200.0e9
nu = 0.3
SCF = 1.0
YS = 355.0e6
thickness = 1.0

# Loop over components of the mesh
for comp in mesh.components:
    comp.set_property('density', density)
    comp.set_property('E', E)
    comp.set_property('nu', nu)
    comp.set_property('SCF', SCF)
    comp.set_property('YS', YS)
    comp.set_property('thickness', thickness)

    # Create stiffness and element object for each component
    stiffness = Stiffness(comp)
    element = Element(comp, stiffness)

# Create a TACS assembler object from the mesh loader
assembler = Assembler(mesh)

# Create a KS function and get the design variable values
ks_func = KSFunction('ks_func.py')
design_vars = ks_func.get_design_vars()

# Get the node locations and create the forces
nodes = mesh.nodes
for node in nodes:
    node.set_property('x', node.get_property('x') + 0.1)
forces = Forces(mesh)

# Set up and solve the analysis problem
u = Vector(len(nodes))
f = Vector(len(nodes))

assembler.assemble(u, f)
A = assembler.factored_matrix()
b = assembler.factored_rhs()
solver = UMFPACKSolver()
solver.solve(A, b, u)

# Evaluate the function and solve for the adjoint variables
obj_func = ks_func.objective_function(u)
adjoint_vars = ks_func.solve_adjoint(obj_func, u)

# Compute the total derivative with respect to material design variables and nodal locations
dobj_dvars = np.zeros((len(design_vars), len(design_vars)))
dobj_dnodes = np.zeros((len(design_vars), len(nodes)))

assembler.compute_total_derivative(dobj_dvars, dobj_dnodes, adjoint_vars)

# Create a random direction along which to perturb the nodes
perturb_dir = np.random.rand(len(nodes))

# Set the complex step and compute the perturbed solution
h = 1.0j * 1e-6
perturbed_u = u + h * perturb_dir

# Evaluate the function for the perturbed solution and compute the projected derivative
perturbed_obj_func = ks_func.objective_function(perturbed_u)
projected_dobj_dnodes = assembler.projected_total_derivative(perturbed_obj_func, perturbed_u, adjoint_vars)

# Output the results for visualization
if rank == 0:
    np.savetxt('dobj_dvars.txt', dobj_dvars)
    np.savetxt('dobj_dnodes.txt', dobj_dnodes)
    np.savetxt('projected_dobj_dnodes.txt', projected_dobj_dnodes)
```

This code assumes that you have a KS function defined in a file named 'ks_func.py'. The KS function should define the objective function and the design variables, and it should implement the methods `get_design_vars()`, `objective_function()`, and `solve_adjoint()`. The KS function should also be compatible with the TACS library.