 ```python
from seedemu.topology import Topology
from seedemu.net import Network, Host, Router
from seedemu.layer import Layer
from seedemu.ebgp import Ebgp

topo = Topology()

# Create base layer
base_layer = Layer(topo)

# Define autonomous systems
as1 = base_layer.addAS(1)
as2 = base_layer.addAS(2)
as3 = base_layer.addAS(3)

# Define networks and routers for AS1
net1 = Network('10.0.0.0/24')
router1 = Router('r1')
host1 = Host('h1', ip=str(net1.getIP(0)))
host2 = Host('h2', ip=str(net1.getIP(1)))
host3 = Host('h3', ip=str(net1.getIP(2)))
host4 = Host('h4', ip=str(net1.getIP(3)))
host5 = Host('h5', ip=str(net1.getIP(4)))
as1.addEntity(net1)
as1.addEntity(router1)
as1.addEntity(host1)
as1.addEntity(host2)
as1.addEntity(host3)
as1.addEntity(host4)
as1.addEntity(host5)

# Define networks and routers for AS2
net2 = Network('10.1.0.0/24')
router2 = Router('r2')
net3 = Network('10.2.0.0/24')
router3 = Router('r3')
net4 = Network('10.3.0.0/24')
router4 = Router('r4')
as2.addEntity(net2)
as2.addEntity(router2)
as2.addEntity(net3)
as2.addEntity(router3)
as2.addEntity(net4)
as2.addEntity(router4)

# Define networks and routers for AS3
net5 = Network('10.4.0.0/24')
router5 = Router('r5')
router6 = Router('r6')
as3.addEntity(net5)
as3.addEntity(router5)
as3.addEntity(router6)

# Create routing layer
routing_layer = Layer(topo)

# Configure interfaces
routing_layer.addInterface(router1.getInterfaceByIndex(0), net1)
routing_layer.addInterface(router2.getInterfaceByIndex(0), net2)
routing_layer.addInterface(router2.getInterfaceByIndex(1), net3)
routing_layer.addInterface(router2.getInterfaceByIndex(2), net4)
routing_layer.addInterface(router3.getInterfaceByIndex(0), net3)
routing_layer.addInterface(router4.getInterfaceByIndex(0), net4)
routing_layer.addInterface(router5.getInterfaceByIndex(0), net5)
routing_layer.addInterface(router6.getInterfaceByIndex(0), net5)

# Create eBGP layer
ebgp_layer = Ebgp(topo)

# Add private peering between ASes
ebgp_layer.addPeer(router1, router2, as_number=1, peer_as_number=2)
ebgp_layer.addPeer(router2, router5, as_number=2, peer_as_number=3)

# Add layers to the emulator
topo.addLayer(base_layer)
topo.addLayer(routing_layer)
topo.addLayer(ebgp_layer)

# Dump emulator state to a binary file
topo.dump('emulation.seed')
```