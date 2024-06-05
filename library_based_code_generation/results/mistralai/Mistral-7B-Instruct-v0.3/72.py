 Here is the Python code that emulates the network as per your instructions, using the `seed-emulator` library and Docker. Note that you need to have `seed-emulator` and `docker` installed on your system.

```python
import seed_emulator as se

# Create the network topology
topology = se.Topology(name='cross-connect')

# Define ASes and their components
as_150 = se.AutonomousSystem(150)
as_151 = se.AutonomousSystem(151)
as_152 = se.AutonomousSystem(152)

web_150 = se.Host(name='web', ip='10.0.0.1', as_number=as_150.as_number)
router0_150 = se.Router(name='router0', ip='10.0.0.2', as_number=as_150.as_number)
web_151 = se.Host(name='web', ip='10.1.0.1', as_number=as_151.as_number)
router0_151 = se.Router(name='router0', ip='10.1.0.2', as_number=as_151.as_number)
web_152 = se.Host(name='web', ip='10.2.0.1', as_number=as_152.as_number)
router0_152 = se.Router(name='router0', ip='10.2.0.2', as_number=as_152.as_number)

# Define networks and their connections
net0_150 = se.Network(name='net0', ipv4_cidr='10.0.0.0/24')
net0_150.add_host(web_150)
net0_150.add_host(router0_150)
net0_151 = se.Network(name='net0', ipv4_cidr='10.1.0.0/24')
net0_151.add_host(web_151)
net0_151.add_host(router0_151)
net0_152 = se.Network(name='net0', ipv4_cidr='10.2.0.0/24')
net0_152.add_host(web_152)
net0_152.add_host(router0_152)

# Connect routers in AS150 and AS152
router0_150.add_connection(router0_152)

# Create internet exchange and peer AS150 and AS151
ix_100 = se.InternetExchange(name='100')
ix_100.add_peer(as_150)
ix_150 = se.InternetExchangeConnection(peer=as_151, local_connection=ix_100.connections[0])
ix_151 = se.InternetExchangeConnection(peer=as_150, local_connection=ix_100.connections[0])

# Set AS150 as a provider for AS152
as_150.set_provider(as_152)

# Add components to the topology
topology.add_autonomous_system(as_150)
topology.add_autonomous_system(as_151)
topology.add_autonomous_system(as_152)
topology.add_host(web_150)
topology.add_host(web_151)
topology.add_host(web_152)
topology.add_router(router0_150)
topology.add_router(router0_151)
topology.add_router(router0_152)
topology.add_network(net0_150)
topology.add_network(net0_151)
topology.add_network(net0_152)
topology.add_internet_exchange(ix_100)
topology.add_internet_exchange_connection(ix_150)
topology.add_internet_exchange_connection(ix_151)

# Render and compile the emulation using Docker with self-managed network
topology.render(dockerfile='Dockerfile', network_driver='bridge')
topology.compile(directory='./cross-connect')
```

This code creates the network topology, adds the required components, and renders and compiles the emulation using Docker with a self-managed network. The compiled emulation will be saved in the directory './cross-connect'.