from seedemu import *

emulator = makeEmulatorBaseWith10StubASAndHosts()

internet_exchange = createInternetExchange(1)
internet_exchange.set_display_name("Internet Exchange 1")
internet_exchange.set_description("Main Internet Exchange")

as1 = createAutonomousSystem(101)
as1.set_display_name("AS 101")
as1.set_description("Autonomous System 101")

as2 = createAutonomousSystem(102)
as2.set_display_name("AS 102")
as2.set_description("Autonomous System 102")

as3 = createAutonomousSystem(103)
as3.set_display_name("AS 103")
as3.set_description("Autonomous System 103")

network1 = as1.create_network("Network 1")
network1.set_display_name("Network 1")
network1.set_description("Network 1 Description")

network2 = as2.create_network("Network 2")
network2.set_display_name("Network 2")
network2.set_description("Network 2 Description")

network3 = as3.create_network("Network 3")
network3.set_display_name("Network 3")
network3.set_description("Network 3 Description")

router1 = as1.create_router("Router 1")
router1.join_networks(network1, network2)

router2 = as2.create_router("Router 2")
router2.join_networks(network2, network3)

router3 = as3.create_router("Router 3")
router3.join_networks(network1, network3)

host1 = as1.create_host("Host 1")
host1.join_network(network1)
host1.install_web_service("Web Service 1")

host2 = as2.create_host("Host 2")
host2.join_network(network2)

host3 = as3.create_host("Host 3")
host3.join_network(network3)

host1.bind_to_host(host2)

as1.peer_with(internet_exchange)
as2.peer_with(internet_exchange)
as3.peer_with(internet_exchange)

emulator.add_layer(internet_exchange)
emulator.add_layer(as1)
emulator.add_layer(as2)
emulator.add_layer(as3)

emulator.render()
emulator.compile_with_docker(enable_internet_map=True)