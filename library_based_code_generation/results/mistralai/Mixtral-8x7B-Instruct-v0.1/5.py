 ```python
from seedemu.internet_exchange import InternetExchange
from seedemu.autonomous_system import AutonomousSystem
from seedemu.network import Network
from seedemu.router import Router
from seedemu.host import Host
from seedemu.service import WebService
from seedemu.topology import Topology

# Initialize the emulator
emulator = Topology()

# Create an Internet Exchange with a specific ID and set its display name and description
ix = emulator.add_internet_exchange('ix-1', 'My Internet Exchange', 'This is my Internet Exchange')

# Create three Autonomous Systems with different IDs
as1 = emulator.add_autonomous_system('as1', 'AS 1', 'This is AS 1')
as2 = emulator.add_autonomous_system('as2', 'AS 2', 'This is AS 2')
as3 = emulator.add_autonomous_system('as3', 'AS 3', 'This is AS 3')

# Create a network, a router, and a host for each Autonomous System
net1 = emulator.add_network(as1, 'net1', 'Network 1 in AS 1')
r1 = emulator.add_router(net1, 'r1', 'Router 1 in AS 1')
h1 = emulator.add_host(net1, 'h1', 'Host 1 in AS 1')

net2 = emulator.add_network(as2, 'net2', 'Network 2 in AS 2')
r2 = emulator.add_router(net2, 'r2', 'Router 2 in AS 2')
h2 = emulator.add_host(net2, 'h2', 'Host 2 in AS 2')

net3 = emulator.add_network(as3, 'net3', 'Network 3 in AS 3')
r3 = emulator.add_router(net3, 'r3', 'Router 3 in AS 3')
h3 = emulator.add_host(net3, 'h3', 'Host 3 in AS 3')

# Install a web service on a virtual node and bind this node to a host
web_service = WebService('web1')
emulator.add_service(h1, web_service)

# Peer the Autonomous Systems with the Internet Exchange
emulator.peer_as(as1, ix)
emulator.peer_as(as2, ix)
emulator.peer_as(as3, ix)

# Add all the layers to the emulator, render the emulator, and compile it with Docker
emulator.add_all_layers()
emulator.render()
emulator.compile(enable_internet_map=True)
```