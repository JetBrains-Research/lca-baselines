  ```
import dpf

# Download multi-stage cyclic result
result = dpf.download_multi_stage_cyclic_result("result_id")

# Create a model from the result
model = dpf.create_model_from_result(result)

# Display the model's state
print(model.state)

# Verify that the model is a multi-stage model
result_info = model.result_info
if result_info.number_of_results > 1:
    print("This is a multi-stage model")
else:
    print("This is not a multi-stage model")

# Get the cyclic support
cyclic_support = model.cyclic_support

# Display the number of stages, sectors in each stage, and nodes in the first stage's base sector
print("Number of stages:", cyclic_support.num_stages)
print("Number of sectors in each stage:", cyclic_support.num_sectors_per_stage)
print("Number of nodes in the first stage's base sector:", cyclic_support.num_nodes_in_base_sector)

# Expand displacement results on chosen sectors
displacement_operator = dpf.cyclic_expanded_displacement(model)
displacement_operator.sectors_to_expand = [1, 2, 3]
displacement_operator.apply()

# Get the total deformation and expanded mesh
deformation = displacement_operator.deformation
mesh = displacement_operator.mesh

# Plot the expanded result on the expanded mesh
plotter = dpf.plotter.Plotter(mesh, deformation)
plotter.plot()

# Expand only some sectors for the mesh
displacement_operator.sectors_to_expand = [1, 3]
displacement_operator.apply()

# Plot the expanded result on the expanded mesh
plotter = dpf.plotter.Plotter(mesh, deformation)
plotter.plot()

# Check results precisely
time_frequency_support = model.time_frequency_support
harmonic_index = time_frequency_support.harmonic_index
print("Harmonic index:", harmonic_index)

# Verify that the displacement values are the same on all nodes
displacement_values = deformation.values
for node in range(displacement_values.shape[0]):
    if not np.allclose(displacement_values[node], displacement_values[node]):
        print("Displacement values are not the same on all nodes")
        break
```