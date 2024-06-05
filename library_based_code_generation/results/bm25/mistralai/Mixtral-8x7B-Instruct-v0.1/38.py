 ```python
from seedemu.lib import makeEmulatorBaseWith10StubASAndHosts, makeEmulatorBaseWith5StubASAndHosts, getNodeByAsnAndName, markAsStub, makeStubAs, makeStubAsWithHosts, make_transit_as, makeTransitAs, PoAServer, WebService, shouldMerge, doesAContainB, make_service_as, getVirtualNodes, registerNodes, up_emulator, _selectImageFor, getBalanceForAll, getNonceForAll, getNodesByName

# Create emulator base
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Create transit AS
transit_as = makeTransitAs(emulator, "transit1", "Transit1", 2, [("ix1", 0.5), ("ix2", 0.5)])
emulator.add_as(transit_as)

# Create stub ASes
stub_as1 = makeStubAsWithHosts(emulator, "stub1", "Stub1", 1, 2, [("host1", "Host1", {"software": "web", "account": "user1"})])
stub_as2 = makeStubAsWithHosts(emulator, "stub2", "Stub2", 1, 2, [("host2", "Host2", {})])
stub_as3 = makeStubAs(emulator, "stub3", "Stub3")
emulator.add_as(stub_as1)
emulator.add_as(stub_as2)
emulator.add_as(stub_as3)

# Connect ASes
emulator.connect_as(transit_as, stub_as1, 0, 3, 0)
emulator.connect_as(transit_as, stub_as2, 1, 3, 1)
emulator.connect_as(stub_as1, stub_as2, 0, 1, 0)

# Create Ebgp layer
ebgp_layer = emulator.add_ebgp_layer(transit_as)

# Create web service layer
web_service_layer = emulator.add_webservice_layer(ebgp_layer)
web_service_node1 = WebService(web_service_layer, "web1", "Web1")
web_service_node2 = WebService(web_service_layer, "web2", "Web2")
web_service_layer.add_node(web_service_node1)
web_service_layer.add_node(web_service_node2)

# Register nodes
registerNodes(emulator)

# Save emulator to component file
emulator.save_to_component_file("emulator.comp")

# Render emulator
emulator.render()

# Change display names for web service nodes
web_service_node1.display_name = "Web Service 1"
web_service_node2.display_name = "Web Service 2"

# Compile emulator using Docker
output_folder = "output"
docker_images = {
    "poa": "my_poa_image",
    "web": "my_web_image",
    "host": "my_host_image",
}
emulator.compile(output_folder, docker_images, _selectImageFor, getBalanceForAll, getNonceForAll, getNodesByName)
```