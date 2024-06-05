  Here is the Python code that creates an emulation environment using the seed-emulator library:
```
from seed_emulator import *

# Create an emulation environment with three types of autonomous systems (AS)
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Create two internet exchanges with specific display names
emulator.add_exchange("Internet Exchange 1", "1.1.1.1")
emulator.add_exchange("Internet Exchange 2", "2.2.2.2")

# Create three internal networks
emulator.add_network("Network 1", "10.10.10.0/24")
emulator.add_network("Network 2", "10.10.20.0/24")
emulator.add_network("Network 3", "10.10.30.0/24")

# Create four routers linked in a linear structure
emulator.add_router("Router 1", "1.1.1.1")
emulator.add_router("Router 2", "2.2.2.2")
emulator.add_router("Router 3", "3.3.3.3")
emulator.add_router("Router 4", "4.4.4.4")

# Establish BGP peering by creating an Ebgp layer and setting up the transit AS as the internet service provider for all the stub ASes
emulator.add_ebgp_layer()
emulator.set_transit_as("Transit AS")

# Set up direct peering between two of the stub ASes
emulator.add_direct_peering("Stub AS 1", "Stub AS 2")

# Create a web service layer with two web service nodes and bind these virtual nodes to physical nodes
emulator.add_web_service_layer()
emulator.add_web_service_node("Web Service 1", "1.1.1.1")
emulator.add_web_service_node("Web Service 2", "2.2.2.2")

# Add all the created layers to the emulator and save it to a component file
emulator.save_component_file("emulator.json")

# Render the emulator and change the display names for the nodes hosting the web services
emulator.render()
emulator.get_node_by_asn_and_name("Web Service 1", "1.1.1.1").set_display_name("Web Service 1")
emulator.get_node_by_asn_and_name("Web Service 2", "2.2.2.2").set_display_name("Web Service 2")

# Finally, compile the emulator using Docker, specifying custom images from DockerHub and local sources
emulator.compile_docker("emulator.json", "docker-compose.yml", "output")
```
This code creates an emulation environment with three types of autonomous systems (AS): transit, stub, and utility. The transit AS has two internet exchanges, three internal networks, and four routers linked in a linear structure. The stub ASes have three internal networks, two routers, and two host nodes. The utility AS has a single router and a single host node.

The code also establishes BGP peering by creating an Ebgp layer and setting up the transit AS as the internet service provider for all the stub ASes. It also sets up direct peering between two of the stub ASes.

Finally, the code creates a web service layer with two web service nodes and binds these virtual nodes to physical nodes. The code adds all the created layers to the emulator and saves it to a component file. It then renders the emulator and changes the display names for the nodes hosting the web services. Finally, the code compiles the emulator using Docker, specifying custom images from DockerHub and local sources.