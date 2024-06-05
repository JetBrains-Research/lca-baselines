 Here is the Python code that emulates the given requirements using the seed-emulator library:

```python
from seed_emulator import Emulator, InternetExchange, PoAServer, Network, Router, AS

# Create an emulator
emulator = Emulator()

# Create AS150 as a transit AS with four routers and three networks
emulator = emulator.makeEmulatorBaseWith10StubASAndHosts(150, 4, 3)

# Create AS151 and AS152 with a router and a web host each
emulator = emulator.makeStubAsWithHosts([(151, 1, 'web_host_151'), (152, 1, 'web_host_152')])

# Get the routers and web hosts for each AS
r150 = emulator.getRouters(150)
r151 = emulator.getNodeByAsnAndName(151, 'router_151')
r152 = emulator.getNodeByAsnAndName(152, 'router_152')
web_host_151 = emulator.getNodes([(151, 'web_host_151')])[0]
web_host_152 = emulator.getNodes([(152, 'web_host_152')])[0]

# Connect AS151 and AS152 to the internet exchange
ie = InternetExchange()
ie.add_member(r151)
ie.add_member(r152)
emulator.setInternetExchange(ie)

# Join AS151 and AS152 to the internet exchange
r151.joinInternetExchange(ie)
r152.joinInternetExchange(ie)

# Set up BGP peering between AS150 and AS151, and between AS150 and AS152
for r in r150:
    r.add_peer(r151, 65001)
    r.add_peer(r152, 65001)

# Join AS151 and AS152 to the networks in AS150
for network in emulator.getNetworks(150):
    r151.joinNetwork(network)
    r152.joinNetwork(network)

# Add all the layers to the emulator
emulator.up_emulator()

# Dump the emulator's state to a binary file
emulator.gen_emulation_files('emulation.json')
```

This code creates an emulation with three autonomous systems (AS150, AS151, and AS152) as described. It sets up BGP peering between AS150 and AS151, and between AS150 and AS152. Finally, it adds all the layers to the emulator and dumps the emulator's state to a binary file named 'emulation.json'.