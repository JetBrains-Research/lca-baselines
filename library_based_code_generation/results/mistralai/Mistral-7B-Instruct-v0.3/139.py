 Here is a Python code sample that demonstrates the use of the pydpf-core library for multi-stage cyclic symmetry analysis with advanced customization. Please note that you need to install the pydpf-core library before running this code.

```python
from pydpf import Model, Result, CyclicOperator, Mesh, Plotter
import numpy as np

# Download a multi-stage cyclic result
result = Result.from_file('path/to/your/result.dat')

# Create a model from the result
model = Model.from_result(result)

# Check if the model is a multi-stage model
if model.is_multi_stage:
    print("Model is a multi-stage model.")

    # Get the result info
    print("Number of stages:", len(model.stages))
    print("Number of sectors in each stage:", [len(stage.sectors) for stage in model.stages])
    print("Number of nodes in the first stage's base sector:", len(model.stages[0].sectors[0].nodes))

    # Expand displacement results on chosen sectors
    displacement_operator = CyclicOperator(operator_type='displacement')
    sectors_to_expand_stage1 = [0, 1]  # Select sectors to expand on the first stage
    sectors_to_expand_stage2 = [1]  # Select sectors to expand on the second stage

    # Expand the displacements and get a total deformation
    total_deformation = displacement_operator.expand(model, sectors_to_expand_stage1, sectors_to_expand_stage2)

    # Get the expanded mesh
    expanded_mesh = Mesh.from_result(total_deformation)

    # Plot the expanded result on the expanded mesh
    Plotter.plot_displacement(total_deformation, expanded_mesh)

    # Demonstrate how to expand only some sectors for the mesh, and plot the expanded result on the expanded mesh
    sectors_to_expand_mesh = [0]  # Select sectors to expand on the mesh
    expanded_mesh_partial = Mesh.from_result(total_deformation, sectors_to_expand_mesh)
    Plotter.plot_displacement(total_deformation, expanded_mesh_partial, sectors_to_expand_mesh)

    # Check results precisely
    print("Time frequency support:", model.time_frequency_support)
    print("Verifying that displacement values are the same on all nodes...")
    for node in expanded_mesh.nodes:
        displacement = total_deformation.get_value(node)
        if np.abs(displacement - total_deformation.get_value(expanded_mesh.nodes[0])) > 1e-12:
            print(f"Displacement values are not the same on all nodes: {displacement} vs {total_deformation.get_value(expanded_mesh.nodes[0])}")
else:
    print("Model is not a multi-stage model.")
```