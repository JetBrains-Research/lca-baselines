 ```python
from seedemu.topology import EmulatorTopology, Network, NetworkType, maskNetwork, updateNetwork, getNetwork, createHostsOnNetwork, getServiceNetwork
from seedemu.services import BgpAttackerComponent
from seedemu.ixnetwork import shouldMerge, mapIxAddress, addIxLink, __get_ix_port, __compileIxNetWorker, __compileIxNetMaster

topology = EmulatorTopology('bgp_attacker')

# Create autonomous systems and networks
as150 = topology.getNodeByAsnAndName('AS150', 'r1')
as150_network = topology.createNetwork('AS150_NETWORK', NetworkType.PRIVATE)
topology.updateNetwork(as150_network, maskNetwork('10.0.0.0/24'))
topology.createHostsOnNetwork(as150_network, 1)

as2 = topology.makeEmulatorBaseWith5StubASAndHosts('AS2', 2, 1)
as2_network = topology.getNetwork('AS2_NETWORK_2')
topology.updateNetwork(as2_network, maskNetwork('20.0.0.0/24'))

as151 = topology.getNodeByAsnAndName('AS151', 'r1')
as151_network = topology.createNetwork('AS151_NETWORK', NetworkType.PRIVATE)
topology.updateNetwork(as151_network, maskNetwork('30.0.0.0/24'))
topology.createHostsOnNetwork(as151_network, 1)

# Create internet exchanges and links
ix100 = topology.getIxByName('ix-100')
ix101 = topology.getIxByName('ix-101')

topology.addIxLink(ix100, as2.getHostByName('h1'), mapIxAddress(ix100, 1))
topology.addIxLink(ix100, as150.getHostByName('h1'), mapIxAddress(ix100, 2))
topology.addIxLink(ix101, as2.getHostByName('h2'), mapIxAddress(ix101, 1))
topology.addIxLink(ix101, as151.getHostByName('h1'), mapIxAddress(ix101, 2))

# Merge internet exchanges
topology.shouldMerge(ix100)
topology.shouldMerge(ix101)

# Compile the emulator topology
topology.__compileIxNetMaster()
topology.__compileIxNetWorker()

# Create BGP attacker component
attacker = BgpAttackerComponent('attacker')
attacker_as = topology.getNodeByAsnAndName('AS151', 'r1')
attacker_network = topology.getNetwork('AS151_NETWORK')
topology.updateNetwork(attacker_network, maskNetwork('30.0.0.0/24'))
topology.createHostsOnNetwork(attacker_network, 1)
attacker_host = attacker_as.getHostByName('h1')
attacker_host.addComponent(attacker)

# Configure private peering relationships
ix100_port = topology.__get_ix_port(ix100)
ix101_port = topology.__get_ix_port(ix101)

topology.joinNetwork(as150_network, as2.getHostByName('h1').getInterfaceByName('eth1'), ix100_port)
topology.joinNetwork(as151_network, as2.getHostByName('h2').getInterfaceByName('eth1'), ix101_port)
topology.joinNetwork(attacker_network, attacker_host.getInterfaceByName('eth1'), ix100_port)

# Render and compile the simulation with Docker
topology.render()
topology.compile()
```