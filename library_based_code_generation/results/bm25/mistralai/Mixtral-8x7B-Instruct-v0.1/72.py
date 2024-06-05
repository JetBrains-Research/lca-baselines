 Sure, here's the Python code based on your instruction:
```python
from seed_emulator.api import shouldMerge, PoAServer, makeEmulatorBaseWith10StubASAndHosts, makeEmulatorBaseWith5StubASAndHosts, doesAContainB, WebService, createEmulation, makeStubAsWithHosts, WebServer, getNodeByAsnAndName, gen_emulation_files, DefaultWebServiceMerger, make_service_as, configureAsEvpnProvider, createHost, getHost, joinInternetExchange, getAsIsds, markAsEdge, makeTransitAs

# Create the emulation base
base_emulation = makeEmulatorBaseWith5StubASAndHosts()

# Create AS150, AS151, and AS152
as150 = makeStubAsWithHosts(base_emulation, "as150", 1)
as151 = makeStubAsWithHosts(base_emulation, "as151", 1)
as152 = makeStubAsWithHosts(base_emulation, "as152", 1)

# Create hosts and web services for each AS
web_hosts = [getNodeByAsnAndName(as, "web") for as in [as150, as151, as152]]
web_services = [WebService(WebServer(80)) for _ in web_hosts]
for host, service in zip(web_hosts, web_services):
    createHost(host, service)

# Create networks for each AS
net0_nodes = [getNodeByAsnAndName(as, "net0") for as in [as150, as151, as152]]
for node in net0_nodes:
    node.addInterface()

# Connect hosts and routers to their respective networks
for asn, host, net in zip([as150, as151, as152], web_hosts, net0_nodes):
    host.addInterface(net)

# Connect AS150 and AS152 routers directly
as150_router = getNodeByAsnAndName(as150, "router0")
as152_router = getNodeByAsnAndName(as152, "router0")
as150_router.connectTo(as152_router)

# Create internet exchange and peer AS150 and AS151
ix100 = PoAServer("ix100", 100)
ix100.addInterface()
as150_router.connectTo(ix100)
as151_router.connectTo(ix100)

# Configure AS150 as a provider for AS152
configureAsEvpnProvider(as150, as152)

# Mark AS150 and AS151 as edge and AS152 as transit
markAsEdge(as150)
markAsEdge(as151)
makeTransitAs(as152)

# Create the emulation
emulation = createEmulation("cross-connect", base_emulation, gen_emulation_files, DefaultWebServiceMerger(), make_service_as())

# Compile the emulation using Docker
emulation.compile(docker_render=True, docker_network="self")
```
This code creates an emulation with three autonomous systems (AS150, AS151, and AS152) and their respective hosts and networks as specified in the instruction. It also sets up the cross-connections and peering relationships between the ASes. Finally, it compiles the emulation using Docker with self-managed network and saves it to the `./cross-connect` directory.