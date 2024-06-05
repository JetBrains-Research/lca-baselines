```python
from seedemu.core import Emulator, InternetExchange, AutonomousSystem
from seedemu.layers import Base, Routing, Ebgp, Ibgp, WebService

# Initialize the emulator and layers
emulator = Emulator()
base = Base()
routing = Routing()
ebgp = Ebgp()
ibgp = Ibgp()
web = WebService()

# Create an Internet Exchange
ix = InternetExchange(100)
ix.setDisplayName("My Internet Exchange")
ix.setDescription("This is a custom Internet Exchange.")
base.addInternetExchange(ix)

# Autonomous Systems details
as_details = [
    (200, "AS200 Network", "AS200 Router", "AS200 Host"),
    (300, "AS300 Network", "AS300 Router", "AS300 Host"),
    (400, "AS400 Network", "AS400 Router", "AS400 Host"),
]

for asn, network_name, router_name, host_name in as_details:
    # Create Autonomous System
    asys = AutonomousSystem(asn)
    asys.setDisplayName(f"Autonomous System {asn}")
    asys.setDescription(f"This is Autonomous System {asn}.")
    
    # Create a network
    network = asys.createNetwork(network_name)
    network.setDisplayName(network_name)
    network.setDescription(f"This is {network_name}.")
    
    # Create a router that joins two networks
    router = asys.createRouter(router_name)
    router.addNetwork(network)
    router.setDisplayName(router_name)
    router.setDescription(f"This is {router_name}.")
    
    # Create a host that joins a network
    host = asys.createHost(host_name)
    host.addNetwork(network)
    host.setDisplayName(host_name)
    host.setDescription(f"This is {host_name}.")
    
    # Install a web service on a virtual node and bind this node to a host
    web.install(host_name)
    
    # Add Autonomous System to the base layer
    base.addAutonomousSystem(asys)
    
    # Peer the Autonomous System with the Internet Exchange
    ebgp.addPeering(ix, asn)

# Add all layers to the emulator
emulator.addLayer(base)
emulator.addLayer(routing)
emulator.addLayer(ebgp)
emulator.addLayer(ibgp)
emulator.addLayer(web)

# Render the emulator
emulator.render()

# Compile with Docker, enabling the internet map
emulator.compile(output="output", enableInternetMap=True)
```