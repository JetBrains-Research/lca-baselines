import seedemu

emulator = seedemu.createEmulation()

base_layer = seedemu.makeEmulatorBaseWith10StubASAndHosts(emulator)

routing_layer = seedemu.createAutonomousSystem(emulator, "Routing")
ebgp_layer = seedemu.createAutonomousSystem(emulator, "Ebgp")
ibgp_layer = seedemu.createAutonomousSystem(emulator, "Ibgp")
ospf_layer = seedemu.createAutonomousSystem(emulator, "Ospf")
webservice_layer = seedemu.createAutonomousSystem(emulator, "WebService")

internet_exchange1 = seedemu.createAutonomousSystem(emulator, "InternetExchange1")
internet_exchange2 = seedemu.createAutonomousSystem(emulator, "InternetExchange2")

transit_as = seedemu.createAutonomousSystem(emulator, "TransitAS")
stub_as = seedemu.createAutonomousSystem(emulator, "StubAS")

host = seedemu.makeStubAsWithHosts(stub_as, "Host1", "192.168.1.10")

real_world_as = seedemu.createAutonomousSystem(emulator, "RealWorldAS")
seedemu.setAutonomousSystem(real_world_as, "RemoteAccess", True)

route_server = seedemu.createAutonomousSystem(emulator, "RouteServer")
seedemu.addPrivatePeering(route_server, transit_as, "Peer1", "Peer2")

seedemu.shouldMerge(base_layer, routing_layer)
seedemu.shouldMerge(base_layer, ebgp_layer)
seedemu.shouldMerge(base_layer, ibgp_layer)
seedemu.shouldMerge(base_layer, ospf_layer)
seedemu.shouldMerge(base_layer, webservice_layer)
seedemu.shouldMerge(base_layer, internet_exchange1)
seedemu.shouldMerge(base_layer, internet_exchange2)
seedemu.shouldMerge(base_layer, transit_as)
seedemu.shouldMerge(base_layer, stub_as)
seedemu.shouldMerge(base_layer, real_world_as)
seedemu.shouldMerge(base_layer, route_server)

seedemu.up_emulator(emulator)
seedemu.toGraphviz(emulator)
seedemu.to_json(emulator)
seedemu.resolveTo(emulator)
seedemu.get_balance_with_name(emulator)
seedemu.createEmulation(emulator)