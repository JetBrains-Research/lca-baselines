```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp, Ibgp, InternetExchange
from seedemu.services import WebService, DomainNameService
from seedemu.compiler import Docker

# Create base layer
base = Base()

# Autonomous Systems
as150 = base.createAutonomousSystem(150)
as151 = base.createAutonomousSystem(151)
as152 = base.createAutonomousSystem(152)

# Networks for AS150
net150_1 = base.createNetwork("net150_1")
net150_2 = base.createNetwork("net150_2")
net150_3 = base.createNetwork("net150_3")

# Routers for AS150
router150_1 = as150.createRouter("router150_1")
router150_2 = as150.createRouter("router150_2")
router150_3 = as150.createRouter("router150_3")
router150_4 = as150.createRouter("router150_4")

# Connect routers to networks in AS150
as150.connect(router150_1, net150_1)
as150.connect(router150_2, net150_2)
as150.connect(router150_3, net150_3)
as150.connect(router150_4, net150_1)  # Example of connecting another router to an existing network

# Web hosts for AS151 and AS152
web151 = as151.createHost("web151")
web152 = as152.createHost("web152")

# Networks for AS151 and AS152
net151 = base.createNetwork("net151")
net152 = base.createNetwork("net152")

# Routers for AS151 and AS152
router151 = as151.createRouter("router151")
router152 = as152.createRouter("router152")

# Connect routers to networks and hosts in AS151 and AS152
as151.connect(router151, net151)
as151.connect(web151, net151)

as152.connect(router152, net152)
as152.connect(web152, net152)

# Routing layer
routing = Routing()

# BGP peering
ebgp = Ebgp()
ebgp.addPeering(150, 151)
ebgp.addPeering(150, 152)

# Internet Exchange
ix = InternetExchange(100)
ix.addMember(151)
ix.addMember(152)

# Add services
web = WebService()
base.addService(web)

# Add all layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(ix)

# Compile and generate the emulation
emu.compile(Docker(outputDirectory="output"))

# Dump the emulator's state to a binary file
emu.dump("emulation_state.bin")
```