 ```python
from seedeme.emulator import Emulator
from seedeme.topology import Topology
from seedeme.autonomous_system import AutonomousSystem
from seedeme.router import Router
from seedeme.network import Network
from seedeme.bgp_peer import BGPPeer
from seedeme.internet_exchange import InternetExchange

# Create Emulator
emulator = Emulator()

# Create ASes
as150 = AutonomousSystem(150)
as151 = AutonomousSystem(151)
as152 = AutonomousSystem(152)

# Create Routers for AS150
r1 = Router("r1")
r2 = Router("r2")
r3 = Router("r3")
r4 = Router("r4")

# Add Routers to AS150
as150.add_router(r1)
as150.add_router(r2)
as150.add_router(r3)
as150.add_router(r4)

# Create Networks for AS150
n1 = Network("10.0.0.0/24")
n2 = Network("10.0.1.0/24")
n3 = Network("10.0.2.0/24")

# Add Networks to AS150
as150.add_network(n1)
as150.add_network(n2)
as150.add_network(n3)

# Create Routers for AS151 and AS152
r5 = Router("r5")
r6 = Router("r6")

# Create Web Hosts for AS151 and AS152
web_host_151 = "10.0.3.100/24"
web_host_152 = "10.0.4.100/24"

# Add Routers to AS151 and AS152
as151.add_router(r5)
as152.add_router(r6)

# Create Networks for AS151 and AS152
n4 = Network("10.0.3.0/24")
n5 = Network("10.0.4.0/24")

# Add Networks to AS151 and AS152
as151.add_network(n4)
as152.add_network(n5)

# Add Web Hosts to AS151 and AS152
as151.add_web_host(web_host_151)
as152.add_web_host(web_host_152)

# Create Internet Exchange
ix = InternetExchange("IX1")

# Add Internet Exchange to AS151 and AS152
as151.add_internet_exchange(ix)
as152.add_internet_exchange(ix)

# Create BGP Peers
peer_1 = BGPPeer(r1, r5, as_external=as151.autonomous_system_number)
peer_2 = BGPPeer(r2, r6, as_external=as152.autonomous_system_number)

# Add BGP Peers to Routers
r1.add_bgp_peer(peer_1)
r5.add_bgp_peer(peer_1)
r2.add_bgp_peer(peer_2)
r6.add_bgp_peer(peer_2)

# Add Layers to Emulator
emulator.add_topology(Topology([as150, as151, as152]))
emulator.add_router_layer(r1)
emulator.add_router_layer(r2)
emulator.add_router_layer(r3)
emulator.add_router_layer(r4)
emulator.add_router_layer(r5)
emulator.add_router_layer(r6)
emulator.add_network_layer(n1)
emulator.add_network_layer(n2)
emulator.add_network_layer(n3)
emulator.add_network_layer(n4)
emulator.add_network_layer(n5)

# Dump Emulator State
emulator.dump("emulation_state.bin")
```