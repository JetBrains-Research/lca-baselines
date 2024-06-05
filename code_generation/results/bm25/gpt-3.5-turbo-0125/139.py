```python
from ansys.dpf import core as dpf

# Download multi-stage cyclic result
result = dpf.download_multi_stage_cyclic_result("path/to/result.rst")

# Create model from result and display state
model = dpf.cyclic_multistage(result)
print(model.state)

# Verify model is multi-stage
result_info = model.result_info
if result_info.get_number_of_results() > 1:
    print("Model is multi-stage")

# Display cyclic support information
num_stages = model.cyclic_support_get_num_sectors()
print("Number of stages:", num_stages)
for stage in range(num_stages):
    num_sectors = model.cyclic_support_get_sectors_scoping(stage).get_length()
    print("Number of sectors in stage", stage, ":", num_sectors)
    if stage == 0:
        num_nodes_base_sector = model.cyclic_support_get_sectors_scoping(stage).get_entity(0).get_length()
        print("Number of nodes in first stage's base sector:", num_nodes_base_sector)

# Expand displacement results on chosen sectors
displacement_operator = dpf.cyclic_expanded_displacement()
displacement_operator.inputs.sectors_to_expand.connect(model.cyclic_support_get_sectors_scoping(0))
displacement_operator.inputs.sectors_to_expand.connect(model.cyclic_support_get_sectors_scoping(1))
expanded_displacements = displacement_operator.get_result()
total_deformation = expanded_displacements.get_total_deformation()
expanded_mesh = expanded_displacements.get_expanded_mesh()

# Plot expanded result on expanded mesh
expanded_mesh.plot(expanded_displacements)

# Expand only some sectors for the mesh
displacement_operator.inputs.sectors_to_expand.connect(model.cyclic_support_get_sectors_scoping(0))
expanded_displacements_some_sectors = displacement_operator.get_result()
expanded_mesh_some_sectors = expanded_displacements_some_sectors.get_expanded_mesh()
expanded_mesh_some_sectors.plot(expanded_displacements_some_sectors)

# Check results precisely
time_frequency_support = expanded_displacements.get_time_frequency_support()
print("Harmonic index:", time_frequency_support.get_harmonic_index())
for node in range(expanded_mesh.get_nodes().get_length()):
    if not expanded_displacements.get_value(node) == expanded_displacements_some_sectors.get_value(node):
        print("Displacement values are not the same on all nodes")
```