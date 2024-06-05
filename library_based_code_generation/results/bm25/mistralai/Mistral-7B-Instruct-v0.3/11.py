 Here is the Python code that emulates the given requirements using the seed-emulator library:

```python
from seed_emulator import Emulator, AutonomousSystem, ScionAutonomousSystem, Network, Host, Router, PoAServer

# Create the emulator
emulator = Emulator()

# Base layer
# Autonomous System 1 (AS1) with 5 hosts and a router
as1 = ScionAutonomousSystem(1)
as1_router = Router("AS1-Router")
as1_network = Network("10.0.0.0/24")
as1_hosts = [Host("AS1-Host{}".format(i)) for i in range(1, 6)]
as1_router.joinNetwork(as1_network)
as1_hosts += [as1_router]
emulator.createAutonomousSystem(as1)
emulator.createNetwork(as1_network)
emulator.createHostsOnNetwork(as1_network, as1_hosts)
emulator.setAutonomousSystem(as1_router, as1)

# Autonomous System 2 (AS2) with 3 routers on different networks
as2 = ScionAutonomousSystem(2)
as2_router1 = Router("AS2-Router1")
as2_router2 = Router("AS2-Router2")
as2_router3 = Router("AS2-Router3")
as2_network1 = Network("172.16.0.0/16")
as2_network2 = Network("172.17.0.0/16")
as2_network3 = Network("172.18.0.0/16")
as2_router1.joinNetwork(as2_network1)
as2_router2.joinNetwork(as2_network2)
as2_router3.joinNetwork(as2_network3)
emulator.createAutonomousSystem(as2)
emulator.createNetwork(as2_network1)
emulator.createNetwork(as2_network2)
emulator.createNetwork(as2_network3)
emulator.createRouter(as2_router1)
emulator.createRouter(as2_router2)
emulator.createRouter(as2_router3)
emulator.setAutonomousSystem(as2_router1, as2)
emulator.setAutonomousSystem(as2_router2, as2)
emulator.setAutonomousSystem(as2_router3, as2)

# Autonomous System 3 (AS3) with 2 routers on the same network
as3 = ScionAutonomousSystem(3)
as3_router1 = Router("AS3-Router1")
as3_router2 = Router("AS3-Router2")
as3_network = Network("192.168.0.0/16")
as3_router1.joinNetwork(as3_network)
as3_router2.joinNetwork(as3_network)
emulator.createAutonomousSystem(as3)
emulator.createNetwork(as3_network)
emulator.createRouter(as3_router1)
emulator.createRouter(as3_router2)
emulator.setAutonomousSystem(as3_router1, as3)
emulator.setAutonomousSystem(as3_router2, as3)

# eBGP layer
# Private peering between different autonomous systems
as1_router_poa = PoAServer(as1_router)
as2_router1_poa = PoAServer(as2_router1)
as2_router2_poa = PoAServer(as2_router2)
as2_router3_poa = PoAServer(as2_router3)
as3_router1_poa = PoAServer(as3_router1)
as3_router2_poa = PoAServer(as3_router2)

as1_router_poa.addPeer(as2_router1_poa)
as1_router_poa.addPeer(as2_router2_poa)
as1_router_poa.addPeer(as2_router3_poa)
as2_router1_poa.addPeer(as3_router1_poa)
as2_router1_poa.addPeer(as3_router2_poa)

# Add all layers to the emulator
emulator.createEmulation()
emulator.setBaseSystem(BaseSystem(emulator.getAutonomousSystem(as1)), as1)
emulator.setBaseSystem(BaseSystem(emulator.getAutonomousSystem(as2)), as2)
emulator.setBaseSystem(BaseSystem(emulator.getAutonomousSystem(as3)), as3)

# Dump the emulator state to a binary file
emulator.dump("emulation.bin")
```