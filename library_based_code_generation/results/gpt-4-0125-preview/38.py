```python
from seedemu import *
from seedemu.layers import Base, Ebgp, Ibgp, Ospf, WebService
from seedemu.services import WebServer, DnsServer, Bind9Server
from seedemu.utilities import Docker

# Base layer setup
base = Base()

# Autonomous Systems
transit_as = base.createAutonomousSystem(100)
stub_as1 = base.createAutonomousSystem(150)
stub_as2 = base.createAutonomousSystem(200)
utility_as = base.createAutonomousSystem(250)

# Transit AS setup
ix1 = base.createInternetExchange(1)
ix1.setDisplayName("IX1")
ix2 = base.createInternetExchange(2)
ix2.setDisplayName("IX2")

transit_as.createNetwork("net0")
transit_as.createNetwork("net1")
transit_as.createNetwork("net2")

r1 = transit_as.createRouter("router1")
r2 = transit_as.createRouter("router2")
r3 = transit_as.createRouter("router3")
r4 = transit_as.createRouter("router4")

r1.joinNetwork("net0")
r2.joinNetwork("net0")
r2.joinNetwork("net1")
r3.joinNetwork("net1")
r3.joinNetwork("net2")
r4.joinNetwork("net2")

# Stub AS setup
net1 = stub_as1.createNetwork("net1")
router1 = stub_as1.createRouter("router1")
host1 = stub_as1.createHost("host1")
host2 = stub_as1.createHost("host2")

router1.joinNetwork("net1")
host1.joinNetwork("net1")
host2.joinNetwork("net1")

host1.addService(WebServer())
host1.addAccount("newuser")

# Second stub AS
net2 = stub_as2.createNetwork("net2")
router2 = stub_as2.createRouter("router2")
host3 = stub_as2.createHost("host3")
host4 = stub_as2.createHost("host4")

router2.joinNetwork("net2")
host3.joinNetwork("net2")
host4.joinNetwork("net2")

# Utility AS setup
# Assuming the utility function is a custom function not defined in the seed-emulator.
# This part will be skipped due to the lack of context on the utility function.

# BGP peering
ebgp = Ebgp()
ebgp.addPeering(transit_as, stub_as1)
ebgp.addPeering(transit_as, stub_as2)
ebgp.addDirectPeering(stub_as1, stub_as2)

# Web service layer
web_service = WebService()
web_service.addWebServer(stub_as1, "host1", "www.example.com")
web_service.addWebServer(stub_as2, "host3", "www.another.com")

# Adding layers to the emulator
emu = Emulator()
emu.addLayer(base)
emu.addLayer(ebgp)
emu.addLayer(web_service)

# Save to a component file
emu.compile(Docker(outputDirectory="output", baseImage="ubuntu:latest"))

# Render the emulator
emu.render()

# Change display names for nodes hosting web services
base.getNodeByName("host1").setDisplayName("WebServer1")
base.getNodeByName("host3").setDisplayName("WebServer2")

# Compile the emulator
emu.compile(Docker(outputDirectory="output", baseImage="ubuntu:latest", customImages={
    "WebServer": "dockerhub_username/web_server_image:latest",
    "Router": "local/router_image:latest"
}))

# Generate Docker files and copy the base container image to the output folder
# This step is automatically handled by the compile method of the Docker utility class.
```