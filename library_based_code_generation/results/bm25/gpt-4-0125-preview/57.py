```python
from seedemu.core import Emulator, Network
from seedemu.layers import Base, Routing, Ebgp, Ibgp, InternetExchange
from seedemu.services import BgpAttacker

# Create base layer
base = Base()

# Create autonomous systems
base.createAutonomousSystem(150)
base.createAutonomousSystem(2)
base.createAutonomousSystem(151)

# Create networks
net150 = base.createNetwork("net150")
net2 = base.createNetwork("net2")
net151 = base.createNetwork("net151")

# Add routers to ASes
base.addNodeToAutonomousSystem(150, "router150")
base.addNodeToAutonomousSystem(2, "router2a")
base.addNodeToAutonomousSystem(2, "router2b")
base.addNodeToAutonomousSystem(151, "router151")

# Add networks to ASes
base.addNetworkToAutonomousSystem(150, net150)
base.addNetworkToAutonomousSystem(2, net2)
base.addNetworkToAutonomousSystem(151, net151)

# Create Internet Exchanges
ix100 = InternetExchange(100)
ix101 = InternetExchange(101)

# Add ASes to IXs
ix100.addMember(150)
ix100.addMember(2)
ix101.addMember(2)
ix101.addMember(151)

# Create BGP attacker
bgpAttacker = BgpAttacker()
bgpAttacker.addPrefixHijack(151, net151)
bgpAttacker.joinInternetExchange(100)

# Create emulator
emu = Emulator()

# Add layers to emulator
emu.addLayer(base)
emu.addLayer(Routing())
emu.addLayer(Ebgp())
emu.addLayer(Ibgp())
emu.addLayer(ix100)
emu.addLayer(ix101)
emu.addLayer(bgpAttacker)

# Compile and render with Docker
emu.render()
emu.compile(Docker())
```