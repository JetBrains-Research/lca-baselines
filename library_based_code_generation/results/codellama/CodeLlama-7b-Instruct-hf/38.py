  Here is the Python code to create the emulation environment using the seed-emulator library:
```
from seed_emulator import Emulator

# Create the emulation environment
emulator = Emulator()

# Create the transit autonomous system
transit_as = emulator.create_as("Transit")

# Create the internet exchanges
transit_as.create_iex("IE1", "IE2")

# Create the internal networks
transit_as.create_network("Network1")
transit_as.create_network("Network2")
transit_as.create_network("Network3")

# Create the routers
transit_as.create_router("Router1")
transit_as.create_router("Router2")
transit_as.create_router("Router3")

# Create the stub autonomous systems
stub_as1 = emulator.create_as("Stub1")
stub_as2 = emulator.create_as("Stub2")
stub_as3 = emulator.create_as("Stub3")

# Create the internal networks for the stub ASes
stub_as1.create_network("Stub1Network")
stub_as2.create_network("Stub2Network")
stub_as3.create_network("Stub3Network")

# Create the routers for the stub ASes
stub_as1.create_router("Stub1Router")
stub_as2.create_router("Stub2Router")
stub_as3.create_router("Stub3Router")

# Create the host nodes for the stub ASes
stub_as1.create_host("Stub1Host1")
stub_as1.create_host("Stub1Host2")
stub_as2.create_host("Stub2Host1")
stub_as2.create_host("Stub2Host2")
stub_as3.create_host("Stub3Host1")

# Establish BGP peering
transit_as.create_ebgp_layer()
transit_as.set_is_isp(True)
stub_as1.set_peer_as(transit_as)
stub_as2.set_peer_as(transit_as)
stub_as3.set_peer_as(transit_as)

# Create the web service layer
web_service_layer = emulator.create_web_service_layer()

# Create the web service nodes
web_service_layer.create_web_service_node("WebService1")
web_service_layer.create_web_service_node("WebService2")

# Bind the web service nodes to physical nodes
web_service_layer.bind_web_service_node("WebService1", "PhysicalNode1")
web_service_layer.bind_web_service_node("WebService2", "PhysicalNode2")

# Add all the created layers to the emulator
emulator.add_layer(transit_as)
emulator.add_layer(stub_as1)
emulator.add_layer(stub_as2)
emulator.add_layer(stub_as3)
emulator.add_layer(web_service_layer)

# Save the emulator to a component file
emulator.save("emulator.component")

# Render the emulator
emulator.render()

# Change the display names for the nodes hosting the web services
web_service_layer.set_display_name("WebService1", "WebService1DisplayName")
web_service_layer.set_display_name("WebService2", "WebService2DisplayName")

# Compile the emulator using Docker
emulator.compile_docker("Dockerfile", "output")

# Generate Docker files and copy the base container image to the output folder
emulator.generate_docker_files("Dockerfile", "output")
emulator.copy_base_container_image("output")
```
This code creates an emulation environment with three types of autonomous systems (AS): transit, stub, and utility. The transit AS has two internet exchanges, three internal networks, and four routers linked in a linear structure. The stub ASes have three internal networks, routers, and host nodes. BGP peering is established between the transit AS and the stub ASes, and direct peering is set up between two of the stub ASes. A web service layer is created with two web service nodes and bound to physical nodes. All the created layers are added to the emulator, and the emulator is saved to a component file. The emulator is then rendered, and the display names for the nodes hosting the web services are changed. Finally, the emulator is compiled using Docker, and the Docker files are generated and copied to the output folder.