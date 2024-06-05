```python
from ansys.dpf import core as dpf
from ansys.dpf.core import examples

# Download multi-stage cyclic result
multistage_cyclic_result = examples.download_multi_stage_cyclic_result()

# Create a model from the result
model = dpf.Model(multistage_cyclic_result)

# Display the model's state
print(model)

# Verify that the model is a multi-stage model
result_info = model.metadata.result_info
if result_info.analysis_type == dpf.AnalysisType.multi_stage_cyclic:
    print("The model is a multi-stage cyclic model.")
else:
    print("The model is not a multi-stage cyclic model.")

# Go over the cyclic support
cyclic_support = model.metadata.cyclic_support
print(f"Number of stages: {cyclic_support.num_stages}")
for stage in range(cyclic_support.num_stages):
    print(f"Stage {stage+1}: {cyclic_support.get_num_sectors(stage)} sectors")
print(f"Number of nodes in the first stage's base sector: {cyclic_support.get_num_nodes_base(0)}")

# Expand displacement results
displacement = model.results.displacement()
cyclic_expansion = displacement.cyclic_expansion()

# Select sectors to expand on the first stage
cyclic_expansion.sectors_to_expand = [0, 1]  # Example sectors

# Select sectors to expand stage by stage
cyclic_expansion.set_expansion_on_all_stages()

# Expand the displacements and get total deformation
total_deformation = cyclic_expansion.eval()

# Get the expanded mesh
expanded_mesh = cyclic_expansion.mesh

# Plot the expanded result on the expanded mesh
total_deformation.plot_contour(mesh=expanded_mesh)

# Expand only some sectors for the mesh
cyclic_expansion.sectors_to_expand = [0]  # Example sector
partial_expanded_mesh = cyclic_expansion.mesh
partial_deformation = cyclic_expansion.eval()

# Plot the expanded result on the partially expanded mesh
partial_deformation.plot_contour(mesh=partial_expanded_mesh)

# Check results precisely
time_freq_support = model.metadata.time_freq_support
print(f"Time frequency support: {time_freq_support}")

# Verify displacement values are the same on all nodes
displacement_values = displacement.on_all_time_freqs.eval()
print(f"Displacement values are consistent across nodes: {displacement_values.is_constant_over_space()}")
```