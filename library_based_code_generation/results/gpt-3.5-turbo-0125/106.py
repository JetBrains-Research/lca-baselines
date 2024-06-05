import seed_emulator

def create_stub_autonomous_system():
    stub_as = seed_emulator.AutonomousSystem()
    web_server = seed_emulator.WebServer()
    router = seed_emulator.Router()
    stub_as.add_device(web_server)
    stub_as.add_device(router)
    return stub_as

base_layer = seed_emulator.BaseLayer()
routing_layer = seed_emulator.RoutingLayer()
ebgp_layer = seed_emulator.EBGPLayer()
ibgp_layer = seed_emulator.IBGPLayer()
ospf_layer = seed_emulator.OSPFLayer()
web_service_layer = seed_emulator.WebServiceLayer()

internet_exchanges = [seed_emulator.InternetExchange() for _ in range(3)]

stub_as_list = [create_stub_autonomous_system() for _ in range(5)]

as1 = seed_emulator.AutonomousSystem()
router1 = seed_emulator.Router()
as1.add_device(router1)
as1.join_network()
as1.join_internet_exchange(internet_exchanges[0])

as2 = seed_emulator.AutonomousSystem()
router2 = seed_emulator.Router()
as2.add_device(router2)
as2.join_network()
as2.join_internet_exchange(internet_exchanges[1])

for asys in stub_as_list:
    asys.join_internet_exchange(internet_exchanges[2])

as1.join_private_peering(as2)

bgp_attacker = seed_emulator.BGPAttacker()
bgp_attacker.hijack_prefixes()
bgp_attacker.join_internet_exchange(internet_exchanges[2])

emulator = seed_emulator.Emulator()
emulator.add_layer(base_layer)
emulator.add_layer(routing_layer)
emulator.add_layer(ebgp_layer)
emulator.add_layer(ibgp_layer)
emulator.add_layer(ospf_layer)
emulator.add_layer(web_service_layer)

emulator.add_autonomous_system(as1)
emulator.add_autonomous_system(as2)
for asys in stub_as_list:
    emulator.add_autonomous_system(asys)

emulator.add_bgp_attacker(bgp_attacker)

emulator.render_emulator()
emulator.compile_emulator(output_directory="specified_directory")