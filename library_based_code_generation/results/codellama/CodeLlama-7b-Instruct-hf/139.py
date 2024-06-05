  import dpf

# Download a multi-stage cyclic result
result = dpf.core.Result('path/to/result.res')

# Create a model from the result
model = result.model

# Display the model's state
print(model.state)

# Verify that the model is a multi-stage model
if model.is_multi_stage:
    print('This is a multi-stage model!')

# Get the number of stages, sectors in each stage, and nodes in the first stage's base sector
num_stages = model.num_stages
num_sectors_per_stage = model.num_sectors_per_stage
num_nodes_in_base_sector = model.num_nodes_in_base_sector

print(f'Number of stages: {num_stages}')
print(f'Number of sectors per stage: {num_sectors_per_stage}')
print(f'Number of nodes in the first stage\'s base sector: {num_nodes_in_base_sector}')

# Expand displacement results on chosen sectors
displacement_operator = dpf.core.Operator('displacement')
displacement_operator.connect(0, model)
displacement_operator.run()

# Select the sectors to expand on the first stage
first_stage_sectors = [0, 1, 2]

# Select the sectors to expand stage by stage
sectors_to_expand = [first_stage_sectors]
for i in range(1, num_stages):
    sectors_to_expand.append(range(i * num_sectors_per_stage, (i + 1) * num_sectors_per_stage))

# Expand the displacements and get a total deformation
deformation = displacement_operator.outputs.fields_container()

# Get the expanded mesh
mesh = model.mesh

# Plot the expanded result on the expanded mesh
import matplotlib.pyplot as plt

plt.figure()
plt.plot(mesh.nodes.coordinates[:, 0], mesh.nodes.coordinates[:, 1], 'bo')
plt.plot(mesh.nodes.coordinates[:, 0], deformation.get_data_at_point(mesh.nodes.indices), 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Expanded Displacement')
plt.show()

# Demonstrate how to expand only some sectors for the mesh
# Create a new mesh with only the selected sectors
new_mesh = mesh.create_subset(sectors_to_expand)

# Expand the displacements on the new mesh
deformation_on_new_mesh = displacement_operator.outputs.fields_container(mesh=new_mesh)

# Plot the expanded result on the new mesh
plt.figure()
plt.plot(new_mesh.nodes.coordinates[:, 0], new_mesh.nodes.coordinates[:, 1], 'bo')
plt.plot(new_mesh.nodes.coordinates[:, 0], deformation_on_new_mesh.get_data_at_point(new_mesh.nodes.indices), 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Expanded Displacement on Selected Sectors')
plt.show()

# Check results precisely
# Print the time frequency support to see the harmonic index
print(deformation.time_frequency_support)

# Verify that the displacement values are the same on all nodes
for i in range(num_nodes_in_base_sector):
    node_displacement = deformation.get_data_at_point(i)
    for j in range(num_stages):
        sector_displacement = deformation.get_data_at_point(i, sector=j)
        if not np.allclose(node_displacement, sector_displacement):
            print(f'Displacement values are not the same on node {i} in sector {j}!')