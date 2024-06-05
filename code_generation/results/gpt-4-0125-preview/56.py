```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Network

# Create base layer
base = Base()

# Create autonomous systems
base.createAutonomousSystem(150)
base.createAutonomousSystem(151)
base.createAutonomousSystem(152)

# Create networks for AS150
net150_1 = base.createNetwork('net-150-1')
net150_2 = base.createNetwork('net-150-2')
net150_3 = base.createNetwork('net-150-3')

# Create routers for AS150
router150_1 = base.createRouter('router-150-1', 150)
router150_2 = base.createRouter('router-150-2', 150)
router150_3 = base.createRouter('router-150-3', 150)
router150_4 = base.createRouter('router-150-4', 150)

# Connect routers to networks in AS150
base.connectRouterToNetwork('router-150-1', 'net-150-1')
base.connectRouterToNetwork('router-150-2', 'net-150-2')
base.connectRouterToNetwork('router-150-3', 'net-150-3')
base.connectRouterToNetwork('router-150-4', 'net-150-1')

# Create networks and routers for AS151 and AS152
net151 = base.createNetwork('net-151')
router151 = base.createRouter('router-151', 151)
base.connectRouterToNetwork('router-151', 'net-151')
base.createHost('web-151', 151).joinNetwork('net-151')

net152 = base.createNetwork('net-152')
router152 = base.createRouter('router-152', 152)
base.connectRouterToNetwork('router-152', 'net-152')
base.createHost('web-152', 152).joinNetwork('net-152')

# Create Internet Exchange and add AS151 and AS152
ix = base.createInternetExchange(200)
ix.join('net-151', 151)
ix.join('net-152', 152)

# Create routing layer
routing = Routing()

# Create BGP layer and configure peering
ebgp = Ebgp()
ebgp.addPrivatePeering(150, 151, 'net-150-1', 'net-151')
ebgp.addPrivatePeering(150, 152, 'net-150-1', 'net-152')

# Create and add all layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump the emulator's state to a binary file
emu.dump('emulation.bin')
```