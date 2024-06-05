 ```python
from seedemu.topology import Topology
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebService
from seedemu.ip import IP
from seedemu.autonomous_system import AutonomousSystem, TransitAS, SingleHomedStubAS, RealWorldAS
from seedemu.peering import PeerRelationship, RouteServerPeer

t = Topology()

base_layer = Base()
t.add_layer(base_layer)

routing_layer = Routing()
t.add_layer(routing_layer)

ebgp_layer = Ebgp()
t.add_layer(ebgp_layer)

ibgp_layer = Ibgp()
t.add_layer(ibgp_layer)

ospf_layer = Ospf()
t.add_layer(ospf_layer)

webservice_layer = WebService()
t.add_layer(webservice_layer)

ix1 = t.add_internet_exchange("IX1")
ix2 = t.add_internet_exchange("IX2")
ix3 = t.add_internet_exchange("IX3")

transit_as1 = TransitAS("TransitAS1")
transit_as2 = TransitAS("TransitAS2")
single_homed_as1 = SingleHomedStubAS("SingleHomedAS1")
single_homed_as2 = SingleHomedStubAS("SingleHomedAS2")
real_world_as = RealWorldAS("RealWorldAS")

t.add_autonomous_system(transit_as1)
t.add_autonomous_system(transit_as2)
t.add_autonomous_system(single_homed_as1)
t.add_autonomous_system(single_homed_as2)
t.add_autonomous_system(real_world_as)

host1 = t.add_host("host1", ip=IP("10.0.0.1"))
transit_as1.add_member(host1)

ebgp_layer.add_peer(transit_as1, transit_as2, PeerRelationship.RS_PEER)
ebgp_layer.add_peer(transit_as2, single_homed_as1, PeerRelationship.PRIVATE_PEER)
ebgp_layer.add_peer(single_homed_as1, single_homed_as2, PeerRelationship.RS_PEER)
ebgp_layer.add_peer(single_homed_as2, real_world_as, PeerRelationship.PRIVATE_PEER)

ebgp_layer.add_route_server(ix1)
ebgp_layer.add_route_server(ix2)
ebgp_layer.add_route_server(ix3)

transit_as1.enable_remote_access()

t.save_components("emulator.comp")
t.render_and_compile()
```