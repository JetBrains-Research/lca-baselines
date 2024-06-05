 ```python
from seedemulator.emulator import Emulator
from seedemulator.topology import Topology
from seedemulator.nodes import (
    InternetExchange,
    AutonomousSystem,
    Router,
    Host,
    Software,
)
from seedemulator.layers import Ebgp, WebService

# Create Emulator
emulator = Emulator()

# Create Transit AS
transit_as = AutonomousSystem(display_name="Transit AS")
emulator.add_autonomous_system(transit_as)

ix1 = InternetExchange(display_name="IX1")
transit_as.add_internet_exchange(ix1)

ix2 = InternetExchange(display_name="IX2")
transit_as.add_internet_exchange(ix2)

net1 = transit_as.add_network("10.0.1.0/24")
net2 = transit_as.add_network("10.0.2.0/24")
net3 = transit_as.add_network("10.0.3.0/24")

router1 = Router(display_name="R1")
transit_as.add_router(router1)
router1.add_interface(net1)

router2 = Router(display_name="R2")
transit_as.add_router(router2)
router2.add_interface(net2)
router2.connect_to(router1)

router3 = Router(display_name="R3")
transit_as.add_router(router3)
router3.add_interface(net3)
router3.connect_to(router2)

router4 = Router(display_name="R4")
transit_as.add_router(router4)
router4.connect_to(router3)

# Create Stub ASes
stub_as1 = AutonomousSystem(display_name="Stub AS1")
emulator.add_autonomous_system(stub_as1)

net4 = stub_as1.add_network("192.168.1.0/24")
router5 = Router(display_name="R5")
stub_as1.add_router(router5)
router5.add_interface(net4)

host1 = Host(display_name="H1")
stub_as1.add_host(host1)
host1.add_interface(net4)
host1.install_software(Software(name="webserver"))
host1.add_account("user1", "password1")

host2 = Host(display_name="H2")
stub_as1.add_host(host2)
host2.add_interface(net4)

stub_as2 = AutonomousSystem(display_name="Stub AS2")
emulator.add_autonomous_system(stub_as2)

net5 = stub_as2.add_network("192.168.2.0/24")
router6 = Router(display_name="R6")
stub_as2.add_router(router6)
router6.add_interface(net5)

host3 = Host(display_name="H3")
stub_as2.add_host(host3)
host3.add_interface(net5)

stub_as3 = AutonomousSystem.create_utility_as(display_name="Stub AS3")
emulator.add_autonomous_system(stub_as3)

# Set up BGP peering
transit_as.add_bgp_peer(stub_as1, Ebgp)
transit_as.add_bgp_peer(stub_as2, Ebgp)
stub_as2.add_bgp_peer(stub_as3, Ebgp)

# Create Web Service Layer
web_service_layer = WebService()
emulator.add_layer(web_service_layer)

node1 = emulator.add_node("webnode1", "node1")
node2 = emulator.add_node("webnode2", "node2")

web_service_layer.bind_node("webnode1", node1)
web_service_layer.bind_node("webnode2", node2)

# Save and Render Emulator
emulator.save_component("emulator_component")
emulator.render("emulator_render", node_display_names={"node1": "WebNode1", "node2": "WebNode2"})

# Compile Emulator
emulator.compile(
    dockerfile="Dockerfile",
    base_image="myregistry/mybaseimage:latest",
    custom_images=[
        {"name": "webserver", "image": "myregistry/webserver:latest"},
        {"name": "bgp", "image": "myregistry/bgp:latest"},
    ],
    output_folder="output",
)
```