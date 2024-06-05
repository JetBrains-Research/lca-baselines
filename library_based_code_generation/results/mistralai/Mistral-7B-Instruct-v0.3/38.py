 ```python
import seed_emulator as se

# Create Transit AS
transit_as = se.AutonomousSystem(name="Transit_AS")
ix1 = transit_as.add_internet_exchange(name="IX1")
ix2 = transit_as.add_internet_exchange(name="IX2")

internal_networks = []
for i in range(1, 4):
    internal_networks.append(transit_as.add_network(f"Internal_{i}"))

routers = []
for i in range(3):
    router = transit_as.add_router(f"Router_{i}")
    if i == 0:
        router.connect(ix1)
    elif i == 2:
        router.connect(ix2)
    for network in internal_networks[i:]:
        router.connect(network)
    if i < 2:
        next_router = routers[i + 1]
        router.connect(next_router)
        routers.append(router)

# Create Stub ASes
stub_as1 = se.AutonomousSystem(name="Stub_AS1")
stub_as2 = se.AutonomousSystem(name="Stub_AS2")
stub_as3 = se.AutonomousSystem(name="Stub_AS3")

# Stub AS1
stub_as1_system1 = stub_as1.add_system(name="System1")
stub_as1_system1_net = stub_as1_system1.add_network(name="Net1")
stub_as1_system1_router = stub_as1_system1.add_router(name="Router1")
stub_as1_system1_router.connect(stub_as1_system1_net)
stub_as1_system1_host1 = stub_as1_system1.add_host(name="Host1")
stub_as1_system1_host1.install_software("Software1")
stub_as1_system1_host1.create_account("Account1")
stub_as1_system1_router.connect(stub_as1_system1_host1)

stub_as1_system2 = stub_as1.add_system(name="System2")
stub_as1_system2_net = stub_as1_system2.add_network(name="Net2")
stub_as1_system2_router = stub_as1_system2.add_router(name="Router2")
stub_as1_system2_router.connect(stub_as1_system2_net)
stub_as1_system2_host1 = stub_as1_system2.add_host(name="Host1")
stub_as1_system2_router.connect(stub_as1_system2_host1)

# Stub AS2
stub_as2_system1 = stub_as2.add_system_from_utility_function(name="System1")
stub_as2_system1.customize()

# Establish BGP peering
ebgp_layer = se.EbgpLayer()
transit_as.add_to_ebgp_layer(ebgp_layer)
for stub_as in [stub_as1, stub_as2, stub_as3]:
    stub_as.add_to_ebgp_layer(ebgp_layer)

stub_as1_router1.peering(stub_as2.routers[0], as_number=stub_as2.as_number)
stub_as2_router1.peering(stub_as3.routers[0], as_number=stub_as3.as_number)

# Create Web Service layer
web_service_layer = se.WebServiceLayer()
web_service_node1 = web_service_layer.add_virtual_node(name="WS_Node1")
web_service_node2 = web_service_layer.add_virtual_node(name="WS_Node2")
physical_node1 = se.PhysicalNode(name="Physical_Node1")
physical_node2 = se.PhysicalNode(name="Physical_Node2")
web_service_node1.bind_to_physical_node(physical_node1)
web_service_node2.bind_to_physical_node(physical_node2)

# Add layers to emulator
emulator = se.Emulator()
emulator.add_layer(transit_as)
emulator.add_layer(stub_as1)
emulator.add_layer(stub_as2)
emulator.add_layer(stub_as3)
emulator.add_layer(ebgp_layer)
emulator.add_layer(web_service_layer)

# Save emulator to component file
emulator.save_to_component_file("emulator.json")

# Render emulator
emulator.render(display_names={f"WS_Node{i}": f"WS_Node_{i}_renamed" for i in range(1, 3)})

# Compile emulator using Docker
dockerfile = se.Dockerfile(emulator)
dockerfile.use("dockerhub/base_image:latest")
dockerfile.use("local_source/custom_image:latest")
dockerfile.save_to_file("Dockerfile")
dockerfile.copy_base_container_image_to_output_folder()
```