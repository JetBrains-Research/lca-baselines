 ```python
import seedemu
from seedemu.layers import Base, Routing, Ebgp

def create_stub_as(asn, exchange):
    # Create hosts and a router for the autonomous system
    hosts = [seedemu.Host(name=f'host{i}') for i in range(1, 4)]
    router = seedemu.Router(name=f'router{asn}')

    # Join hosts to a network
    network = seedemu.Network(name=f'network{asn}')
    for host in hosts:
        host.add_interface(network.get_interface())

    # Join the router to the network and the exchange
    router.add_interface(network.get_interface())
    router.add_interface(exchange.get_interface())

    return router, hosts, network

# Create the emulation
emulation = seedemu.Emulation(name='my_emulation')

# Create the Base layer
base_layer = Base(emulation)
emulation.add_layer(base_layer)

# Create the Routing layer
routing_layer = Routing(emulation)
emulation.add_layer(routing_layer)

# Create the Ebgp layer
ebgp_layer = Ebgp(emulation)
emulation.add_layer(ebgp_layer)

# Create autonomous systems
as1 = seedemu.AutonomousSystem(asn=100, routers=[create_stub_as(100, ebgp_layer.exchanges[0])[0]])
as2 = seedemu.AutonomousSystem(asn=200, routers=[create_stub_as(200, ebgp_layer.exchanges[1])[0]])
as3 = seedemu.AutonomousSystem(asn=300, routers=[create_stub_as(300, ebgp_layer.exchanges[2])[0]])

# Add private peering relationships between different autonomous systems
as1.add_peer(as2.routers[0])
as2.add_peer(as3.routers[0])

# Dump the emulator state to a binary file
emulation.dump('my_emulation.seedemu')
```