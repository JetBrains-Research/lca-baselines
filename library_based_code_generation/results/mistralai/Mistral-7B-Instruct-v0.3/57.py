 Here is the Python code using the `seedemu` library to create the network topology as per your instructions. I've also included the BGP attacker component and the private peering relationships. However, please note that Docker configuration and compilation are beyond the scope of this platform and would require additional steps.

```python
from seedemu import Network, Router, Link, AutonomousSystem, InternetExchange, BGP, BGPAttacker

# Create autonomous systems
as150 = AutonomousSystem(150)
as2 = AutonomousSystem(2)
as151 = AutonomousSystem(151)

# Create routers and networks
r1_as150 = Router(name='r1_as150')
r2_as150 = Router(name='r2_as150')
r1_as2 = Router(name='r1_as2')
r2_as2 = Router(name='r2_as2')
r1_as151 = Router(name='r1_as151')

network_as150 = Network(as150, '10.0.0.0/24')
network_as2 = Network(as2, '192.168.0.0/24')
network_as151 = Network(as151, '172.16.0.0/24')

# Connect routers and networks
link_r1_as150_r2_as150 = Link(r1_as150, r2_as150)
link_r1_as150_network_as150 = Link(r1_as150, network_as150)
link_r2_as150_network_as150 = Link(r2_as150, network_as150)

link_r1_as2_r2_as2 = Link(r1_as2, r2_as2)
link_r1_as2_network_as2 = Link(r1_as2, network_as2)
link_r2_as2_network_as2 = Link(r2_as2, network_as2)

link_r1_as151_network_as151 = Link(r1_as151, network_as151)

# Create internet exchanges and connect ASes
ix_100 = InternetExchange(name='IX 100')
ix_101 = InternetExchange(name='IX 101')

ix_link_as150_ix_100 = Link(ix_100, r1_as150)
ix_link_as2_ix_100 = Link(ix_100, r1_as2)
ix_link_attacker_ix_100 = Link(ix_100, BGPAttacker(name='attacker'))

ix_link_as2_ix_101 = Link(ix_101, r1_as2)
ix_link_as151_ix_101 = Link(ix_101, r1_as151)

# Configure BGP and peering relationships
bgp_as150 = BGP(r1_as150)
bgp_as2 = BGP(r1_as2)
bgp_attacker = BGP(attacker)

bgp_as150.add_peer(bgp_as2, ix_link_as150_ix_100)
bgp_as2.add_peer(bgp_as150, ix_link_as150_ix_100)
bgp_as2.add_peer(bgp_attacker, ix_link_attacker_ix_100)
bgp_attacker.hijack_prefix(network_as151)

bgp_as2.add_peer(bgp_as151, ix_link_as2_ix_101)
bgp_as151.add_peer(bgp_as2, ix_link_as2_ix_101)

# Create network topology
network = Network(as1=as150, as2=as2, as3=as151, routers=[r1_as150, r2_as150, r1_as2, r2_as2, r1_as151], links=[
    link_r1_as150_r2_as150, link_r1_as150_network_as150, link_r2_as150_network_as150,
    link_r1_as2_r2_as2, link_r1_as2_network_as2, link_r2_as2_network_as2,
    link_r1_as151_network_as151,
    ix_link_as150_ix_100, ix_link_as2_ix_100, ix_link_attacker_ix_100,
    ix_link_as2_ix_101, ix_link_as151_ix_101
])
```

For Docker configuration and compilation, you would need to create a Dockerfile that installs seedemu and its dependencies, copies the Python script, and sets up the environment for running the simulation. After that, you can build and run the Docker image. This process is specific to your development environment and is not included in the provided code.