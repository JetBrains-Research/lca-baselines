import seedemu

def create_stub_as(asn, exchange):
    as_stub = seedemu.AutonomousSystem(asn)
    router = seedemu.Router()
    as_stub.add_router(router)
    for i in range(5):
        host = seedemu.Host()
        as_stub.add_host(host)
        router.add_interface(host)
    router.join_network(seedemu.Network())
    router.join_exchange(exchange)
    return as_stub

base_layer = seedemu.Layer("Base")
routing_layer = seedemu.Layer("Routing")
ebgp_layer = seedemu.Layer("Ebgp")

as1 = create_stub_as(100, seedemu.Exchange())
as2 = create_stub_as(200, seedemu.Exchange())
as3 = create_stub_as(300, seedemu.Exchange())

as1.add_private_peering(as2)
as2.add_private_peering(as3)
as3.add_private_peering(as1)

seedemu.add_layer(base_layer)
seedemu.add_layer(routing_layer)
seedemu.add_layer(ebgp_layer)

seedemu.dump_state("emulator_state.bin")