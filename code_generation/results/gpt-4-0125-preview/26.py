```python
from seedemu.core import Emulator, Scenario
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.services import WebServer, DnsServer
from seedemu.hooks import ResolvConfHook

# Create the emulator
emulator = Emulator()

# Create layers
base = Base()
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
ospf = Ospf()
webservice = WebService()

# Create Internet Exchanges
ix100 = base.createInternetExchange(100, displayName="IX100 - Main Exchange")
ix200 = base.createInternetExchange(200, displayName="IX200 - Secondary Exchange")

# Create Transit AS
transit_as1 = base.createAutonomousSystem(65001)
transit_as2 = base.createAutonomousSystem(65002)

# Create single-homed stub AS
stub_as1 = base.createAutonomousSystem(65100)
stub_as2 = base.createAutonomousSystem(65101)

# Add services to AS
webservice.install("65100")
webservice.install("65101")

# Add a host with a customized IP address
base.createHost("webserver1", parent="65100").addService(WebServer())
base.getHostByName("webserver1").setAddress("10.0.0.1/24")

# Create real-world AS
real_as = base.createAutonomousSystem(65003, stub=False, real=True)

# Enable remote access
base.enableRemoteAccess("65100")

# Setup peering via a route server
ebgp.addRouteServer(ix100, 65001)
ebgp.addRouteServerClient(ix100, 65001, 65100)
ebgp.addRouteServerClient(ix100, 65001, 65101)

# Setup private peering
ebgp.addPrivatePeering(ix200, 65002, 65003, peer1Asn=65002, peer2Asn=65003)

# Add layers to the emulator
emulator.addLayer(base)
emulator.addLayer(routing)
emulator.addLayer(ebgp)
emulator.addLayer(ibgp)
emulator.addLayer(ospf)
emulator.addLayer(webservice)

# Save the emulator to a component file
emulator.save("emulation_environment.json")

# Render and compile the emulator
emulator.render()
emulator.compile()
```