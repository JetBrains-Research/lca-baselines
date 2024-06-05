 ```python
from seedemu import Emulator, InternetExchange, AutonomousSystem, Network, Router, Host, WebService, VirtualNode, Container

emulator = Emulator()

ix = InternetExchange(id='IX-1', display_name='My Internet Exchange', description='A custom Internet Exchange')
emulator.add_layer(ix)

as1 = AutonomousSystem(id='AS1', display_name='ASN1', description='Autonomous System 1')
as2 = AutonomousSystem(id='AS2', display_name='ASN2', description='Autonomous System 2')
as3 = AutonomousSystem(id='AS3', display_name='ASN3', description='Autonomous System 3')
emulator.add_layer(as1)
emulator.add_layer(as2)
emulator.add_layer(as3)

# Networks
net1 = Network(id='NET1', display_name='Network 1', description='Network for AS1')
net2 = Network(id='NET2', display_name='Network 2', description='Network for AS2')
net3 = Network(id='NET3', display_name='Network 3', description='Network for AS3')
as1.add_network(net1)
as2.add_network(net2)
as3.add_network(net3)

# Routers
r1 = Router(id='R1', display_name='Router 1', description='Router connecting AS1 and AS2')
r2 = Router(id='R2', display_name='Router 2', description='Router connecting AS1 and AS3')
r3 = Router(id='R3', display_name='Router 3', description='Router connecting AS2 and AS3')
net1.add_router(r1)
net2.add_router(r2)
net3.add_router(r3)
r1.connect(r2)
r2.connect(r3)

# Hosts
h1 = Host(id='H1', display_name='Host 1', description='Host in AS1')
h2 = Host(id='H2', display_name='Host 2', description='Host in AS2')
h3 = Host(id='H3', display_name='Host 3', description='Host in AS3')
net1.add_host(h1)
net2.add_host(h2)
net3.add_host(h3)

# Web service
ws = WebService(id='WS', display_name='Web Service', description='A web service on a virtual node')
vn = VirtualNode(id='VN', display_name='Virtual Node', description='A virtual node for the web service')
vn.install(ws)
h1.bind(vn)

# Peer Autonomous Systems with the Internet Exchange
as1.peer(ix)
as2.peer(ix)
as3.peer(ix)

# Add all layers to the emulator
emulator.add_layer(h1)
emulator.add_layer(h2)
emulator.add_layer(h3)

# Render and compile the emulator with Docker
emulator.render()
emulator.compile(docker=True)
```