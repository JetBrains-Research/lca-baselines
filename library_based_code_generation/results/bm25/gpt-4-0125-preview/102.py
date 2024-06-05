```python
from seedemu import *

def create_stub_as(emulator, asn, exchange):
    asys = emulator.createAutonomousSystem(asn)
    router = asys.createRouter('router1')
    network = asys.createNetwork('net1')
    for i in range(1, 3):
        host = asys.createHost(f'host{i}')
        asys.joinNetwork(f'net1', host)
    asys.joinNetwork('net1', router)
    emulator.joinInternetExchange(asn, exchange, router)

emulator = Emulator()

base = Base()
routing = Routing()
ebgp = Ebgp()

# Create Internet Exchanges
ix100 = emulator.createInternetExchange(100)
ix200 = emulator.createInternetExchange(200)

# Create Autonomous Systems
create_stub_as(emulator, 1, 100)
create_stub_as(emulator, 2, 100)
create_stub_as(emulator, 3, 200)
create_stub_as(emulator, 4, 200)

# Add private peering
ebgp.addPrivatePeering(1, 2, '10.0.0.1', '10.0.0.2')
ebgp.addPrivatePeering(3, 4, '10.0.1.1', '10.0.1.2')

# Add layers to emulator
emulator.addLayer(base)
emulator.addLayer(routing)
emulator.addLayer(ebgp)

# Dump emulator state
emulator.dump('emulator_state.bin')
```