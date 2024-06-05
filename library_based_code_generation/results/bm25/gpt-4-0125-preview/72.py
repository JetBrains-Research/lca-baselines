```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Service, Docker
from seedemu.services import WebService, WebServer
from seedemu.utilities import Docker as DockerUtility

# Create base layer
base = Base()

# Create autonomous systems
as150 = base.createAutonomousSystem(150)
as151 = base.createAutonomousSystem(151)
as152 = base.createAutonomousSystem(152)

# Create networks and nodes
for asn in [150, 151, 152]:
    base.createNetwork(f'net0', asn)
    base.createHost(f'web', asn).joinNetwork(f'net0')
    base.createRouter(f'router0', asn).joinNetwork(f'net0')

# Install web service on 'web' hosts
web_service = WebService()
for asn in [150, 151, 152]:
    web_service.install(f'web@AS{asn}')

# Create routing and eBGP layers
routing = Routing()
ebgp = Ebgp()

# Cross connect AS150 and AS152
ebgp.addPrivatePeerings(150, [('router0', 152, 'router0')])

# Create Internet Exchange and add peers
ix100 = base.createInternetExchange(100)
ix100.addMember(150, 'router0')
ix100.addMember(151, 'router0')

# AS150 as a provider for AS152
ebgp.addProviderCustomer(150, 152)

# Compile the emulation
emu = createEmulation()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(Service().addService(web_service))
emu.addLayer(Docker())

# Render and compile with Docker
DockerUtility(emulation=emu, outputDirectory='./cross-connect').render()
```