from seedemu import Emulator, InternetExchange, AutonomousSystem, Network, Router, Host, VirtualNode

emulator = Emulator()

ix = InternetExchange(1)
ix.display_name = "Internet Exchange"
ix.description = "Main Internet Exchange Point"

as1 = AutonomousSystem(101)
as2 = AutonomousSystem(102)
as3 = AutonomousSystem(103)

network1 = Network("Network 1")
network2 = Network("Network 2")
network3 = Network("Network 3")

router1 = Router("Router 1")
router2 = Router("Router 2")
router3 = Router("Router 3")

host1 = Host("Host 1")
host2 = Host("Host 2")
host3 = Host("Host 3")

virtual_node = VirtualNode("Web Service Node")

network1.add_device(router1)
network1.add_device(host1)

network2.add_device(router2)
network2.add_device(host2)

network3.add_device(router3)
network3.add_device(host3)

virtual_node.install_web_service()
virtual_node.bind_to_host(host1)

as1.add_network(network1)
as2.add_network(network2)
as3.add_network(network3)

as1.peer_with(ix)
as2.peer_with(ix)
as3.peer_with(ix)

emulator.add_layer(ix)
emulator.add_layer(as1)
emulator.add_layer(as2)
emulator.add_layer(as3)

emulator.render()
emulator.compile_with_docker(enable_internet_map=True)