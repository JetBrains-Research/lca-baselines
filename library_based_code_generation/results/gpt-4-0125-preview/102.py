```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp
from seedemu.services import *
from seedemu.core import Emulator, Binder

def create_stub_as(base, asn, exchange):
    asys = base.createAutonomousSystem(asn)
    router = asys.createRouter('router')
    host = asys.createHost('host')
    network = asys.createNetwork('net')
    network.attach(router, host)
    exchange.attach('router', asn)

# Create the base layer
base = Base()

# Create the routing layer
routing = Routing()

# Create the Ebgp layer
ebgp = Ebgp()

# Create Internet Exchanges
ix100 = base.createInternetExchange(100)
ix200 = base.createInternetExchange(200)

# Create autonomous systems and add them to exchanges
create_stub_as(base, 1, ix100)
create_stub_as(base, 2, ix100)
create_stub_as(base, 3, ix200)
create_stub_as(base, 4, ix200)

# Add private peering relationships
ebgp.addPrivatePeering(1, 2, 100)
ebgp.addPrivatePeering(3, 4, 200)

# Create the emulator
emulator = Emulator()

# Add the layers to the emulator
emulator.addLayer(base)
emulator.addLayer(routing)
emulator.addLayer(ebgp)

# Dump the emulator state to a binary file
emulator.dump('emulation.bin')
```