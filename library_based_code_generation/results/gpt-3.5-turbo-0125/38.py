import seed_emulator

# Create transit AS
transit_as = seed_emulator.AutonomousSystem(display_name="Transit AS")
ix1 = seed_emulator.InternetExchange(display_name="IX1")
ix2 = seed_emulator.InternetExchange(display_name="IX2")
network1 = seed_emulator.InternalNetwork()
network2 = seed_emulator.InternalNetwork()
network3 = seed_emulator.InternalNetwork()
router1 = seed_emulator.Router()
router2 = seed_emulator.Router()
router3 = seed_emulator.Router()
router4 = seed_emulator.Router()
transit_as.add_internet_exchange(ix1)
transit_as.add_internet_exchange(ix2)
transit_as.add_internal_network(network1)
transit_as.add_internal_network(network2)
transit_as.add_internal_network(network3)
transit_as.add_router(router1)
transit_as.add_router(router2)
transit_as.add_router(router3)
transit_as.add_router(router4)
transit_as.link_routers([router1, router2, router3, router4])

# Create stub AS
stub_as1 = seed_emulator.AutonomousSystem(display_name="Stub AS1")
stub_as2 = seed_emulator.AutonomousSystem(display_name="Stub AS2")
stub_as3 = seed_emulator.AutonomousSystem(display_name="Stub AS3")
network_stub1 = seed_emulator.InternalNetwork()
network_stub2 = seed_emulator.InternalNetwork()
router_stub1 = seed_emulator.Router()
router_stub2 = seed_emulator.Router()
router_stub3 = seed_emulator.Router()
router_stub4 = seed_emulator.Router()
host1 = seed_emulator.HostNode()
host2 = seed_emulator.HostNode()
host3 = seed_emulator.HostNode()
stub_as1.add_internal_network(network_stub1)
stub_as1.add_router(router_stub1)
stub_as1.add_host_node(host1)
stub_as1.add_host_node(host2)
stub_as2.add_internal_network(network_stub2)
stub_as2.add_router(router_stub2)
stub_as2.add_host_node(host3)
stub_as3 = seed_emulator.create_utility_as()

# Establish BGP peering
ebgp_layer = seed_emulator.EbgpLayer()
ebgp_layer.set_isp(transit_as)
ebgp_layer.add_customer(stub_as1)
ebgp_layer.add_customer(stub_as2)
ebgp_layer.add_customer(stub_as3)
ebgp_layer.set_direct_peering(stub_as1, stub_as2)

# Create web service layer
web_service_node1 = seed_emulator.WebServiceNode()
web_service_node2 = seed_emulator.WebServiceNode()
web_service_layer = seed_emulator.WebServiceLayer()
web_service_layer.add_web_service_node(web_service_node1)
web_service_layer.add_web_service_node(web_service_node2)
web_service_layer.bind_virtual_to_physical()

# Add layers to emulator
emulator = seed_emulator.Emulator()
emulator.add_layer(transit_as)
emulator.add_layer(stub_as1)
emulator.add_layer(stub_as2)
emulator.add_layer(stub_as3)
emulator.add_layer(ebgp_layer)
emulator.add_layer(web_service_layer)

# Save emulator to component file
emulator.save_to_file("emulator_component.json")

# Render emulator
emulator.render()
emulator.change_display_name(web_service_node1, "New Display Name 1")
emulator.change_display_name(web_service_node2, "New Display Name 2")

# Compile emulator using Docker
docker_compiler = seed_emulator.DockerCompiler()
docker_compiler.set_custom_images(["image1", "image2"])
docker_compiler.set_local_sources(["source1", "source2"])
docker_compiler.generate_docker_files()
docker_compiler.copy_base_image("output_folder")