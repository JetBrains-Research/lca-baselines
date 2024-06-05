```python
from seedemu.core import Emulator, Binding, Filter
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.services import WebServer, BgpAttacker
from seedemu.utilities import DockerImageCompiler

def create_stub_as(base_layer, web_service_layer, asn, exchange, prefix, web_content):
    # Create a stub AS with a web server and a router
    base_layer.createAutonomousSystem(asn)
    base_layer.createRouter(f'router-{asn}')
    base_layer.createHost(f'web-{asn}', asys=asn)
    base_layer.createLink(f'link-{asn}', nodes=[f'router-{asn}', f'web-{asn}'])
    web_service_layer.install(f'web-{asn}', WebServer(web_content))
    base_layer.addPrefixToAutonomousSystem(asn, prefix)
    exchange.join(f'router-{asn}', asn)

# Initialize emulator and layers
emulator = Emulator()
base_layer = Base()
routing_layer = Routing()
ebgp_layer = Ebgp()
ibgp_layer = Ibgp()
ospf_layer = Ospf()
web_service_layer = WebService()

# Create Internet Exchanges
ix100 = base_layer.createInternetExchange(100)
ix200 = base_layer.createInternetExchange(200)
ix300 = base_layer.createInternetExchange(300)

# Create stub ASes and join them to Internet Exchanges
create_stub_as(base_layer, web_service_layer, 65001, ix100, '10.0.1.0/24', 'Content for AS65001')
create_stub_as(base_layer, web_service_layer, 65002, ix200, '10.0.2.0/24', 'Content for AS65002')
create_stub_as(base_layer, web_service_layer, 65003, ix300, '10.0.3.0/24', 'Content for AS65003')

# Create ASes with routers joining different networks and Internet Exchanges
base_layer.createAutonomousSystem(65010)
base_layer.createRouter('router-65010')
ix100.join('router-65010', 65010)
ix200.join('router-65010', 65010)

base_layer.createAutonomousSystem(65020)
base_layer.createRouter('router-65020')
ix200.join('router-65020', 65020)
ix300.join('router-65020', 65020)

# Define private peerings
ebgp_layer.addPrivatePeering(65001, 65010, '10.0.100.0/30')
ebgp_layer.addPrivatePeering(65002, 65020, '10.0.200.0/30')

# Add BGP attacker
attacker_asn = 666
base_layer.createAutonomousSystem(attacker_asn)
base_layer.createRouter('attacker-router')
ix300.join('attacker-router', attacker_asn)
ebgp_layer.install('attacker-router', BgpAttacker(prefixes=['10.0.1.0/24'], asn=attacker_asn))

# Merge layers into emulator
emulator.addLayer(base_layer)
emulator.addLayer(routing_layer)
emulator.addLayer(ebgp_layer)
emulator.addLayer(ibgp_layer)
emulator.addLayer(ospf_layer)
emulator.addLayer(web_service_layer)

# Render the emulator
emulator.render()

# Compile with Docker
compiler = DockerImageCompiler(emulator, './output')
compiler.compile()
```