```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp

# Create the base layer
base = Base()

# Autonomous System 1
as1 = base.createAutonomousSystem(100)
net100 = base.createNetwork('net100')
router_as1 = as1.createRouter('router_as1')
as1.createHostsOnNetwork('net100', 5)
router_as1.joinNetwork(net100)

# Autonomous System 2
as2 = base.createAutonomousSystem(200)
net200 = base.createNetwork('net200')
net201 = base.createNetwork('net201')
net202 = base.createNetwork('net202')
router_as2_1 = as2.createRouter('router_as2_1')
router_as2_2 = as2.createRouter('router_as2_2')
router_as2_3 = as2.createRouter('router_as2_3')
router_as2_1.joinNetwork(net200)
router_as2_2.joinNetwork(net201)
router_as2_3.joinNetwork(net202)

# Autonomous System 3
as3 = base.createAutonomousSystem(300)
net300 = base.createNetwork('net300')
router_as3_1 = as3.createRouter('router_as3_1')
router_as3_2 = as3.createRouter('router_as3_2')
router_as3_1.joinNetwork(net300)
router_as3_2.joinNetwork(net300)

# Create the routing layer
routing = Routing()

# Create the eBGP layer
ebgp = Ebgp()

# Add private peering between ASes
ebgp.addPrivatePeering(100, 'router_as1', 200, 'router_as2_1', 'net100')
ebgp.addPrivatePeering(200, 'router_as2_2', 300, 'router_as3_1', 'net201')

# Create the emulation
emu = Emulation()

# Add layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump the emulator state to a binary file
emu.dump('emulation.bin')
```