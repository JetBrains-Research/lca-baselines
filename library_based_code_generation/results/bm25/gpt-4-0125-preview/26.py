```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.services import WebServer, DnsServer
from seedemu.hooks import ResolvConfHook

# Create base layer
base = Base()

# Create layers
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
ospf = Ospf()
webservice = WebService()

# Create Internet Exchanges
ix100 = base.createInternetExchange(100, displayName="IX100")
ix200 = base.createInternetExchange(200, displayName="IX200")
ix300 = base.createInternetExchange(300, displayName="IX300")

# Create Transit AS
transit_as1 = base.createAutonomousSystem(65001)
transit_as2 = base.createAutonomousSystem(65002)

# Create single-homed stub AS with a web service
stub_as1 = base.createAutonomousSystem(65100)
stub_as2 = base.createAutonomousSystem(65101)
webservice.install("65100")
webservice.install("65101")

# Add a host with a customized IP address to one of the AS
base.addHost("webhost", "65100", ipAddresses=["192.168.100.2"])

# Create real-world AS
real_as = base.createAutonomousSystem(65003, stub=False, real=True)

# Enable remote access
base.addHost("remote-access-host", "65003", remoteAccess=True)

# Setup peering via a route server
ebgp.addRsPeering(100, 65001, 65100)
ebgp.addRsPeering(100, 65002, 65101)

# Setup private peering
ebgp.addPrivatePeering(200, 65001, 65002, "65001:65002")

# Add all layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(ibgp)
emu.addLayer(ospf)
emu.addLayer(webservice)

# Save the emulator to a component file
emu.compile()

# Render and compile the emulator
emu.render()
emu.compile()
```