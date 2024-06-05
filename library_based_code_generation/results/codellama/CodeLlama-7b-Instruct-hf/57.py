  Here is the Python code to create the network topology using the seedemu library:
```
from seedemu import Network, Router, NetworkInterface, BGPAttacker

# Create the network topology
network = Network()

# Create the autonomous systems
as150 = network.add_as("AS150")
as2 = network.add_as("AS2")
as151 = network.add_as("AS151")

# Create the routers and networks
router1 = as150.add_router("Router1")
router2 = as2.add_router("Router2")
router3 = as151.add_router("Router3")

network1 = as150.add_network("Network1")
network2 = as2.add_network("Network2")
network3 = as151.add_network("Network3")

# Create the internet exchanges
ix100 = network.add_ix("IX100")
ix101 = network.add_ix("IX101")

# Connect the routers to the internet exchanges
router1.add_interface(network1, ix100)
router2.add_interface(network2, ix100)
router3.add_interface(network3, ix101)

# Create the BGP attacker component
attacker = BGPAttacker(as151, "Attacker")

# Merge the attacker component with the main simulation
network.merge_component(attacker)

# Establish private peering relationships
as150.add_peering(as2, ix100)
as151.add_peering(as2, ix101)
attacker.add_peering(as2, ix100)

# Render and compile the simulation with Docker
network.render_and_compile("docker")
```
This code creates a network topology with three autonomous systems (AS) and three routers, each connected to an internet exchange (IX). The AS150 and AS2 have one router and one network each, while AS151 has two routers and one network. The AS150 and AS2 are connected through IX 100, and AS2 and AS151 are connected through IX 101.

The BGP attacker component is created and merged with the main simulation. The attacker component hijacks the prefix of AS151 and joins IX 100.

Finally, private peering relationships are established between AS150 and AS2 at IX 100, between AS151 and AS2 at IX 101, and between the attacker and AS2 at IX 100. The simulation is rendered and compiled with Docker, managing the network internally.