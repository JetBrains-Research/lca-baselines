 ```
from seedemu.lib import shouldMerge, makeEmulatorBaseWith5StubASAndHosts, makeEmulatorBaseWith10StubASAndHosts, getNodeByAsnAndName, PoAServer, doesAContainB, joinInternetExchange, joinNetwork, __joinNetwork, createEmulation, getNetworks, getRouters, __get_xc_routers, getMaskedNetworks, gen_emulation_files, setInternetExchange, makeStubAsWithHosts, up_emulator, InternetExchange, get_all_accounts_with_node_info

# Create emulation base
emulationBase = makeEmulatorBaseWith10StubASAndHosts()

# Create AS150, AS151, AS152
as150 = makeStubAsWithHosts(emulationBase, "AS150", 4, 3)
as151 = makeStubAsWithHosts(emulationBase, "AS151", 1, 1)
as152 = makeStubAsWithHosts(emulationBase, "AS152", 1, 1)

# Create networks and routers
net1 = as150.getRouters()[0].__joinNetwork(as150.getNetworks()[0], "10.0.0.1")
net2 = as150.getRouters()[1].__joinNetwork(as150.getNetworks()[1], "10.0.1.1")
net3 = as150.getRouters()[2].__joinNetwork(as150.getNetworks()[2], "10.0.2.1")
net4 = as150.getRouters()[3].__joinNetwork(as150.getNetworks()[3], "10.0.3.1")

router151 = as151.getRouters()[0]
router152 = as152.getRouters()[0]

webHost151 = as151.getHosts()[0]
webHost152 = as152.getHosts()[0]

# Add networks to emulation base
emulationBase.addNetwork(net1)
emulationBase.addNetwork(net2)
emulationBase.addNetwork(net3)
emulationBase.addNetwork(net4)

# Create internet exchange
ix = InternetExchange("IX1")
emulationBase.addNode(ix)

# Add routers to internet exchange
joinInternetExchange(router151, ix)
joinInternetExchange(router152, ix)

# Set up BGP peering
as150.getRouters()[0].peerWith(as151.getRouters()[0], "10.0.0.2", "10.0.0.1")
as150.getRouters()[1].peerWith(as151.getRouters()[0], "10.0.1.2", "10.0.0.1")
as150.getRouters()[2].peerWith(as152.getRouters()[0], "10.0.2.2", "10.0.0.1")
as150.getRouters()[3].peerWith(as152.getRouters()[0], "10.0.3.2", "10.0.0.1")

# Add all layers to the emulator
emulationBase.addNode(as150)
emulationBase.addNode(as151)
emulationBase.addNode(as152)
emulationBase.addNode(webHost151)
emulationBase.addNode(webHost152)

# Dump the emulator's state to a binary file
gen_emulation_files(emulationBase, "emulation_state.bin")
```