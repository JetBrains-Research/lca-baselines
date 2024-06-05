```python
from seedemu.core import Emulator, Binding, Environment
from seedemu.layers import Base, Routing, Ebgp, Ibgp, WebService, Docker
from seedemu.services import WebServer, WebClient
from seedemu.hooks import ResolvConfHook

# Create a new instance of the emulator
emulator = Emulator()

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

# Create and install web service on each web host
web_service = WebService()
web_service.install('web@150')
web_service.install('web@151')
web_service.install('web@152')

# Create routing layer
routing = Routing()

# Create BGP layer
ebgp = Ebgp()
ibgp = Ibgp()

# Cross connect AS150 and AS152
routing.addCrossConnection(('router0@150', 'router0@152'))

# Create an Internet Exchange and add peers
ix100 = base.createInternetExchange(100)
ix100.addMember('router0@150')
ix100.addMember('router0@151')

# AS150 is a provider for AS152
ebgp.addProviderCustomer('router0@150', 'router0@152')

# Compile the emulation
emulator.addLayer(base)
emulator.addLayer(routing)
emulator.addLayer(ebgp)
emulator.addLayer(ibgp)
emulator.addLayer(web_service)
emulator.addLayer(Docker())

# Set the environment to use self-managed network
emulator.setEnvironment(Environment(SelfManagedNetwork=True))

# Render and compile the emulation
emulator.render()
emulator.compile(Docker(), './cross-connect')
```