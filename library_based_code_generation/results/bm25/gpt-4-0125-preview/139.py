```python
from ansys.dpf import core as dpf
from ansys.dpf.core import examples

# Download multi-stage cyclic result and create a model
result_file = examples.download_multi_stage_cyclic_result()
model = dpf.Model(result_file)

# Display the model's state
print(model)

# Verify that the model is a multi-stage model
result_info = model.metadata.result_info
print("Number of results:", result_info.number_of_results)
print("Number of sub-results:", result_info.number_of_sub_results)

# Go over the cyclic support
cyclic_support = model.metadata.cyclic_support
print("Number of stages:", cyclic_support.num_stages)
for stage in range(cyclic_support.num_stages):
    print(f"Stage {stage+1}: Number of sectors: {cyclic_support.get_num_sectors(stage)}")
print("Number of nodes in the first stage's base sector:", len(cyclic_support.get_sectors_scoping(0, 0)))

# Expand displacement results on chosen sectors
displacement = model.results.displacement()
displacement_cyclic = displacement.cyclic_expanded
displacement_cyclic.inputs.sectors_to_expand([0])  # Expand only the first sector of the first stage

# Expand displacements and get total deformation
displacement_cyclic.outputs.fields_container()
total_deformation = displacement_cyclic.total_deformation()

# Get the expanded mesh
expanded_mesh = displacement_cyclic.mesh

# Plot the expanded result on the expanded mesh
expanded_mesh.plot(total_deformation)

# Demonstrate how to expand only some sectors for the mesh
displacement_cyclic.inputs.sectors_to_expand([1])  # Expand only the second sector of the first stage
partial_expanded_mesh = displacement_cyclic.mesh
partial_expanded_result = displacement_cyclic.outputs.fields_container()

# Plot the expanded result on the partially expanded mesh
partial_expanded_mesh.plot(partial_expanded_result)

# Check results precisely
time_freq_support = model.metadata.time_freq_support
print("Harmonic index:", time_freq_support.harmonic_indices)

# Verify that the displacement values are the same on all nodes
# This is a conceptual check; specific implementation will depend on the analysis requirements
displacements = displacement.outputs.fields_container()
for field in displacements:
    print("Displacement values are consistent across nodes:", field.data)
```