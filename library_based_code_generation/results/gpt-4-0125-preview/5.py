```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp, Ibgp, WebService, InternetExchange

# Create the base layer
base = Base()

# Create Internet Exchange
ix100 = InternetExchange(100)
ix100.setDisplayName("IX100")
ix100.setDescription("Internet Exchange 100")

# Autonomous System 1
as200 = base.createAutonomousSystem(200)
as200.setDisplayName("AS200")
as200.setDescription("Autonomous System 200")
net200 = as200.createNetwork("net200")
net200.setDisplayName("Network 200")
net200.setDescription("Network in AS200")
router200 = as200.createRouter("router200")
router200.setDisplayName("Router 200")
router200.setDescription("Router in AS200")
router200.joinNetwork("net200")
router200.joinNetwork("ix100")
host200 = as200.createHost("host200")
host200.setDisplayName("Host 200")
host200.setDescription("Host in AS200")
host200.joinNetwork("net200")

# Autonomous System 2
as300 = base.createAutonomousSystem(300)
as300.setDisplayName("AS300")
as300.setDescription("Autonomous System 300")
net300 = as300.createNetwork("net300")
net300.setDisplayName("Network 300")
net300.setDescription("Network in AS300")
router300 = as300.createRouter("router300")
router300.setDisplayName("Router 300")
router300.setDescription("Router in AS300")
router300.joinNetwork("net300")
router300.joinNetwork("ix100")
host300 = as300.createHost("host300")
host300.setDisplayName("Host 300")
host300.setDescription("Host in AS300")
host300.joinNetwork("net300")

# Autonomous System 3
as400 = base.createAutonomousSystem(400)
as400.setDisplayName("AS400")
as400.setDescription("Autonomous System 400")
net400 = as400.createNetwork("net400")
net400.setDisplayName("Network 400")
net400.setDescription("Network in AS400")
router400 = as400.createRouter("router400")
router400.setDisplayName("Router 400")
router400.setDescription("Router in AS400")
router400.joinNetwork("net400")
router400.joinNetwork("ix100")
host400 = as400.createHost("host400")
host400.setDisplayName("Host 400")
host400.setDescription("Host in AS400")
host400.joinNetwork("net400")

# Web service
web = WebService()
web.install("host200")
web.install("host300")
web.install("host400")

# Peering with Internet Exchange
ebgp = Ebgp()
ebgp.addInternetExchangePeering(100, 200)
ebgp.addInternetExchangePeering(100, 300)
ebgp.addInternetExchangePeering(100, 400)

# Add layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(Routing())
emu.addLayer(Ibgp())
emu.addLayer(ebgp)
emu.addLayer(web)
emu.addLayer(ix100)

# Render and compile
emu.render()
emu.compile(Docker(), enableInternetMap=True)
```