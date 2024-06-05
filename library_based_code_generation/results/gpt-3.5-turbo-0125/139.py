import pydpf_core

# Download multi-stage cyclic result
result = pydpf_core.download_multi_stage_cyclic_result()

# Create model from result
model = pydpf_core.create_model(result)

# Display model state
model.display_state()

# Verify model is multi-stage
result_info = model.get_result_info()
if result_info.is_multi_stage_model():
    print("Model is a multi-stage model")

# Display cyclic support
num_stages = result_info.get_num_stages()
for stage in range(num_stages):
    num_sectors = result_info.get_num_sectors_in_stage(stage)
    num_nodes = result_info.get_num_nodes_in_sector(stage, 0)
    print(f"Stage {stage+1}: {num_sectors} sectors, {num_nodes} nodes in base sector")

# Expand displacement results
displacement_operator = pydpf_core.create_displacement_cyclic_operator()
sectors_to_expand = [0]  # Select sectors to expand on first stage
for stage in range(num_stages):
    sectors_to_expand = [sector for sector in range(result_info.get_num_sectors_in_stage(stage))]
    displacement_operator.select_sectors_to_expand(sectors_to_expand)
    displacement_operator.expand_displacements()
    total_deformation = displacement_operator.get_total_deformation()
    expanded_mesh = displacement_operator.get_expanded_mesh()

# Plot expanded result on expanded mesh
pydpf_core.plot_result(expanded_mesh, total_deformation)

# Expand only some sectors for the mesh
sectors_to_expand = [0, 1]  # Select sectors to expand
displacement_operator.select_sectors_to_expand(sectors_to_expand)
displacement_operator.expand_displacements()
total_deformation = displacement_operator.get_total_deformation()
expanded_mesh = displacement_operator.get_expanded_mesh()

# Plot expanded result on expanded mesh
pydpf_core.plot_result(expanded_mesh, total_deformation)

# Check results precisely
time_frequency_support = result_info.get_time_frequency_support()
print(f"Time frequency support: {time_frequency_support}")

# Verify displacement values are the same on all nodes
if displacement_operator.verify_displacement_values():
    print("Displacement values are the same on all nodes")