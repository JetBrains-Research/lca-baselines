  from seedemu import Emulator, Layer, Host, Router, AutonomousSystem, InternetExchange, PrivatePeeringRelationship

# Create the emulator
emulator = Emulator()

# Create the base layer
base_layer = Layer("Base")
emulator.add_layer(base_layer)

# Create the routing layer
routing_layer = Layer("Routing")
emulator.add_layer(routing_layer)

# Create the eBGP layer
ebgp_layer = Layer("eBGP")
emulator.add_layer(ebgp_layer)

# Create multiple autonomous systems
as1 = AutonomousSystem(1)
as2 = AutonomousSystem(2)
as3 = AutonomousSystem(3)

# Create hosts and routers for each autonomous system
host1 = Host("host1", as1)
host2 = Host("host2", as1)
host3 = Host("host3", as2)
host4 = Host("host4", as2)
host5 = Host("host5", as3)
host6 = Host("host6", as3)

router1 = Router("router1", as1)
router2 = Router("router2", as1)
router3 = Router("router3", as2)
router4 = Router("router4", as2)
router5 = Router("router5", as3)
router6 = Router("router6", as3)

# Join the hosts and routers to the networks
base_layer.add_host(host1)
base_layer.add_host(host2)
base_layer.add_host(host3)
base_layer.add_host(host4)
base_layer.add_host(host5)
base_layer.add_host(host6)

routing_layer.add_router(router1)
routing_layer.add_router(router2)
routing_layer.add_router(router3)
routing_layer.add_router(router4)
routing_layer.add_router(router5)
routing_layer.add_router(router6)

# Join the routers to the internet exchanges
ie1 = InternetExchange("ie1")
ie2 = InternetExchange("ie2")
ie3 = InternetExchange("ie3")

router1.join_exchange(ie1)
router2.join_exchange(ie1)
router3.join_exchange(ie2)
router4.join_exchange(ie2)
router5.join_exchange(ie3)
router6.join_exchange(ie3)

# Add private peering relationships between the autonomous systems
as1.add_private_peering_relationship(as2)
as1.add_private_peering_relationship(as3)
as2.add_private_peering_relationship(as1)
as2.add_private_peering_relationship(as3)
as3.add_private_peering_relationship(as1)
as3.add_private_peering_relationship(as2)

# Add the layers to the emulator
emulator.add_layer(base_layer)
emulator.add_layer(routing_layer)
emulator.add_layer(ebgp_layer)

# Dump the emulator state to a binary file
emulator.dump_state("emulator_state.bin")