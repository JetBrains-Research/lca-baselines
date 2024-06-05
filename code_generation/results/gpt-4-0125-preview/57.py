```python
from seedemu.core import Emulator, BindingScope
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Network
from seedemu.services import BgpRouter, WebService, DnsService
from seedemu.hooks import ResolvConfHook
from seedemu.compiler import Docker

# Create a new instance of the emulator
emu = Emulator()

# Create base layer
base = Base()

# Add autonomous systems
base.addAs(150)
base.addAs(2)
base.addAs(151)

# Add networks to ASes
base.addNetwork('net0', 150)
base.addNetwork('net1', 2)
base.addNetwork('net2', 151)

# Add routers to ASes
base.addRouter('router0', 150)
base.addRouter('router1', 2)
base.addRouter('router2', 2)
base.addRouter('router3', 151)

# Connect routers to networks
base.connect('router0', 'net0')
base.connect('router1', 'net1')
base.connect('router2', 'net1')
base.connect('router3', 'net2')

# Create routing layer
routing = Routing()

# Create and configure BGP routers
bgp = Ebgp()
bgp.addPrivatePeering(100, 'router0', 'router1')
bgp.addPrivatePeering(101, 'router3', 'router2')

# Add internet exchanges
ix100 = bgp.addInternetExchange(100)
ix101 = bgp.addInternetExchange(101)

# Connect ASes to internet exchanges
ix100.addMember(150)
ix100.addMember(2)
ix101.addMember(2)
ix101.addMember(151)

# Create BGP attacker
bgp.addAttacker('attacker', 100, 'net2')

# Establish private peering relationships
bgp.addPrivatePeering(100, 'attacker', 'router1')

# Merge layers into the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(bgp)

# Compile the simulation with Docker
compiler = Docker()
compiler.compile(emu, './output')
```