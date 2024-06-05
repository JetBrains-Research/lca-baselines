import seedemu

# Create emulation environment
env = seedemu.Environment()

# Create layers
base_layer = env.create_layer("Base")
routing_layer = env.create_layer("Routing")
ebgp_layer = env.create_layer("Ebgp")
ibgp_layer = env.create_layer("Ibgp")
ospf_layer = env.create_layer("Ospf")
webservice_layer = env.create_layer("WebService")

# Create Internet Exchanges
ix1 = env.create_internet_exchange("IX1", display_name="Internet Exchange 1")
ix2 = env.create_internet_exchange("IX2", display_name="Internet Exchange 2")

# Create Transit Autonomous Systems
transit_as1 = env.create_transit_as("TransitAS1")
transit_as2 = env.create_transit_as("TransitAS2")

# Create single-homed stub Autonomous Systems
stub_as1 = env.create_stub_as("StubAS1")
stub_as2 = env.create_stub_as("StubAS2")

# Add host to Autonomous System
host1 = env.create_host("Host1", ip_address="192.168.1.100")
stub_as1.add_host(host1)

# Create real-world Autonomous System
real_as = env.create_real_as("RealAS")

# Enable remote access to Autonomous System network
real_as.enable_remote_access()

# Set up peering via route server
env.set_up_peering("RouteServer", transit_as1, transit_as2)

# Set up private peering with different peer relationships
env.set_up_private_peering("Peer1", transit_as1, stub_as1, relationship="Customer")
env.set_up_private_peering("Peer2", transit_as2, stub_as2, relationship="Provider")

# Add layers to emulator
env.add_layer(base_layer)
env.add_layer(routing_layer)
env.add_layer(ebgp_layer)
env.add_layer(ibgp_layer)
env.add_layer(ospf_layer)
env.add_layer(webservice_layer)

# Save emulator to component file
env.save_to_file("emulation_environment.json")

# Render and compile emulator
env.render()
env.compile()