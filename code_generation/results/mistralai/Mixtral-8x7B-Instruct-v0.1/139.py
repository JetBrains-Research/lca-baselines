 import pydpf
import pydpf.core as pydpf_core
import pydpf.plot as pydpf_plot
import pydpf.results as pydpf_results
import pydpf.mesh as pydpf_mesh

# Download multi-stage cyclic result
result = pydpf_core.download_result('https://my-api.com/download/result')

# Create model from result
model = pydpf_core.Model(result)

# Display model state
print(model.state)

# Verify that the model is a multi-stage model
assert model.result_info['model_type'] == 'multi-stage'

# Get cyclic support
print('Number of stages:', model.cyclic_support.n_stages)
print('Number of sectors in each stage:', [len(stage) for stage in model.cyclic_support.stages])
print('Number of nodes in the first stage base sector:', len(model.cyclic_support.base_sector(1)))

# Expand displacement results on chosen sectors
displacement_op = pydpf_core.DisplacementCyclicOperator(model)
sectors_to_expand_stage1 = [0, 1]
sectors_to_expand_stage2 = [0]
displacement_op.expand(sectors_to_expand_stage1, sectors_to_expand_stage2)

# Get total deformation
total_deformation = displacement_op.get_total_deformation()

# Get expanded mesh
expanded_mesh = pydpf_mesh.extract_expanded_mesh(model, displacement_op)

# Plot expanded result on expanded mesh
pydpf_plot.plot_expanded_result(expanded_mesh, total_deformation)

# Expand only some sectors for the mesh
sectors_to_expand_mesh = [0]
expanded_mesh_subset = pydpf_mesh.extract_expanded_mesh_subset(model, displacement_op, sectors_to_expand_mesh)

# Plot expanded result on expanded mesh subset
pydpf_plot.plot_expanded_result(expanded_mesh_subset, total_deformation)

# Check results precisely
print('Time frequency support:', pydpf_results.get_time_frequency_support(result))

for node in model.nodes:
    node_displacements = displacement_op.get_displacements(node)
    assert all(abs(displacements[0] - displacement) < 1e-8 for displacement in displacements[1:]), 'Displacements are not the same on all nodes'