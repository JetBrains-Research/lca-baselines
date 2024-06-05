```python
from seedemu.core import Emulator, Binding, Filter
from seedemu.layers import Base, Routing, Ebgp
from seedemu.services import DomainNameCachingService, DnsCachingServer

# Create the base layer
base = Base()

# Create autonomous systems and internet exchanges
asns = [100, 200, 300]
ixps = [1, 2]

for asn in asns:
    base.createAutonomousSystem(asn)
    router = base.createRouter(asn)
    for host_id in range(1, 4):
        host = base.createHost(asn)
        base.createNetwork(asn).addHost(host).addRouter(router)

for ixp in ixps:
    base.createInternetExchangePoint(ixp)
    for asn in asns:
        base.getRouter(asn).joinInternetExchangePoint(ixp)

# Create the routing layer
routing = Routing()

# Create the eBGP layer
ebgp = Ebgp()

# Add private peerings between ASes
ebgp.addPrivatePeering(100, 200, 1)
ebgp.addPrivatePeering(200, 300, 2)

# Create the domain name caching service
dns = DomainNameCachingService()

# Install DNS caching servers on specific hosts
for asn in asns:
    host = base.getHosts(asn)[0]  # Assuming the first host for simplicity
    dns.install(host)

# Add bindings for the DNS service
bindings = Binding()
bindings.addBinding(Filter(asn=100), DnsCachingServer())

# Create the emulator and add all the layers
emu = Emulator()
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)
emu.addLayer(dns)
emu.addBinding(bindings)

# Dump the state of the emulator to a binary file
emu.dump('emulation_state.bin')
```