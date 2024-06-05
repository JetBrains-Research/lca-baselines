  from seed_emulator import Emulator

emulator = Emulator()

# Define a function to create a stub autonomous system with a web server and a router
def create_stub_as(name, network, internet_exchange, web_server=False):
    as_ = emulator.create_autonomous_system(name)
    router = as_.create_router()
    if web_server:
        web_server = as_.create_web_server()
    network.add_router(router)
    internet_exchange.add_router(router)
    return as_

# Create three internet exchanges
ie1 = emulator.create_internet_exchange("IE1")
ie2 = emulator.create_internet_exchange("IE2")
ie3 = emulator.create_internet_exchange("IE3")

# Create multiple stub autonomous systems that join these exchanges
as1 = create_stub_as("AS1", emulator.create_network("Network1"), ie1)
as2 = create_stub_as("AS2", emulator.create_network("Network2"), ie2)
as3 = create_stub_as("AS3", emulator.create_network("Network3"), ie3)

# Create two autonomous systems with routers that join different networks and internet exchanges
as4 = create_stub_as("AS4", emulator.create_network("Network4"), ie1)
as5 = create_stub_as("AS5", emulator.create_network("Network5"), ie2)

# Define private peerings between different autonomous systems
as1.create_private_peering(as2)
as2.create_private_peering(as3)
as3.create_private_peering(as4)
as4.create_private_peering(as5)

# Add a BGP attacker component that hijacks certain prefixes and joins an internet exchange
attacker = emulator.create_bgp_attacker("Attacker")
attacker.add_hijacked_prefix("10.0.0.0/8")
ie1.add_router(attacker)

# Merge the BGP attacker with the emulator and render the new emulator
emulator.merge_components(attacker)

# Compile the new emulator using Docker and output the result to a specified directory
emulator.compile("output_directory")