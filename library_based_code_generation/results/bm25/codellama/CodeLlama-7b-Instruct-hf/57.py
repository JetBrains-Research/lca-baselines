  Here is the Python code that creates the network topology using the seedemu library:
```
from seedemu import *

# Create the network topology
network = createNetwork(NetworkType.IPv4, "my_network")

# Add the autonomous systems
as150 = network.addAutonomousSystem(150, "AS150")
as2 = network.addAutonomousSystem(2, "AS2")
as151 = network.addAutonomousSystem(151, "AS151")

# Add the routers
router1 = as150.addRouter("router1")
router2 = as2.addRouter("router2")
router3 = as151.addRouter("router3")

# Add the networks
network1 = as150.addNetwork("network1")
network2 = as2.addNetwork("network2")
network3 = as151.addNetwork("network3")

# Add the internet exchanges
ix100 = network.addInternetExchange("ix100")
ix101 = network.addInternetExchange("ix101")

# Connect the autonomous systems through the internet exchanges
as150.addIxLink(ix100)
as2.addIxLink(ix100)
as2.addIxLink(ix101)
as151.addIxLink(ix101)

# Create the BGP attacker component
attacker = BgpAttackerComponent("attacker")

# Merge the attacker with the main simulation
shouldMerge(attacker, network)

# Establish private peering relationships
as150.addPrivatePeering(as2, ix100)
as151.addPrivatePeering(as2, ix101)
attacker.addPrivatePeering(as2, ix100)

# Render and compile the simulation with Docker
network.renderAndCompile(docker=True)
```
This code creates a network topology with three autonomous systems (AS) and two internet exchanges (IX). The AS150 and AS2 have one router and one network each, while AS151 has two routers and one network. The AS150 and AS2 are connected through IX 100, and AS2 and AS151 are connected through IX 101. The BGP attacker component is created and merged with the main simulation. Private peering relationships are established between the AS150 and AS2 at IX 100, between the AS151 and AS2 at IX 101, and between the attacker and AS2 at IX 100. Finally, the simulation is rendered and compiled with Docker.