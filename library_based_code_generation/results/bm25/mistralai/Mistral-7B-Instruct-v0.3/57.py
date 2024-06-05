 Here is the Python code that creates the network topology using the seedemu library, creates a BGP attacker component, and merges it with the main simulation. The code also establishes private peering relationships and renders and compiles the simulation with Docker.

```python
from seedemu.core import EmulatorBase, Network, NetworkType, maskNetwork, updateNetwork, getNetwork, __joinNetwork, joinNetwork, createHostsOnNetwork, getServiceNetwork
from seedemu.ix import mapIxAddress, addIxLink, BgpAttackerComponent
from seedemu.docker import DockerNetwork

# Create autonomous systems
AS150 = EmulatorBase.makeEmulatorBaseWith1StubASAndHosts(asn=150, name='AS150')
AS2 = EmulatorBase.makeEmulatorBaseWith5StubASAndHosts(asn=2, name='AS2')
AS151 = EmulatorBase.makeEmulatorBaseWith1StubASAndHosts(asn=151, name='AS151')

# Create networks and routers
AS150_NETWORK = Network(NetworkType.IPV4, maskNetwork('10.0.0.0/24'))
AS2_NETWORK = Network(NetworkType.IPV4, maskNetwork('192.168.0.0/24'))
AS151_NETWORK = Network(NetworkType.IPV4, maskNetwork('172.16.0.0/24'))

AS150.getNetwork(AS150_NETWORK.asn).update(AS150_NETWORK)
AS2.getNetwork(AS2_NETWORK.asn).update(AS2_NETWORK)
AS151.getNetwork(AS151_NETWORK.asn).update(AS151_NETWORK)

AS150_ROUTER = AS150.getNodesByAsnAndName(150)[0]
AS2_ROUTER1 = AS2.getNodesByAsnAndName(2)[0]
AS2_ROUTER2 = AS2.getNodesByAsnAndName(2)[1]
AS151_ROUTER = AS151.getNodesByAsnAndName(151)[0]

# Create internet exchanges and links
IX100 = mapIxAddress('IX100')
IX101 = mapIxAddress('IX101')

AS150_ROUTER.addLink(__get_ix_port(IX100), AS2_ROUTER1, IX100)
AS2_ROUTER1.addLink(__get_ix_port(IX100), AS151_ROUTER, IX101)
AS2_ROUTER2.addLink(__get_ix_port(IX101), AS151_ROUTER, IX101)

# Create BGP attacker component
attacker = BgpAttackerComponent(AS2_ROUTER1, AS151_NETWORK.prefix, AS151_NETWORK.prefix + '/32')

# Merge the attacker component with the main simulation
AS2.merge(attacker)

# Establish private peering relationships
AS150_ROUTER.addLink(__get_ix_port(IX100), AS2_ROUTER1, IX100, peering=True)
AS2_ROUTER2.addLink(__get_ix_port(IX101), AS151_ROUTER, IX101, peering=True)

# Create hosts on networks
AS150_HOST = createHostsOnNetwork(AS150_NETWORK, 1)
AS2_HOST1 = createHostsOnNetwork(AS2_NETWORK, 1)
AS2_HOST2 = createHostsOnNetwork(AS2_NETWORK, 1)
AS151_HOST = createHostsOnNetwork(AS151_NETWORK, 1)

# Render and compile the simulation with Docker
DockerNetwork(AS150, AS2, AS151, AS150_HOST, AS2_HOST1, AS2_HOST2, AS151_HOST).renderAndCompile()
```

Please note that this code assumes you have the seedemu library installed and properly configured. Also, the code does not include error handling or logging, which you may want to add in a production environment.

Regarding the Docker part, you'll need to have Docker installed on your machine and the seedemu Docker image available. If not, you can build the image using the provided Dockerfile in the seedemu repository.