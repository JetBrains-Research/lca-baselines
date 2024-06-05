```python
from seedemu import *
from seedemu.layers import *

# Create a new emulation environment
emulator = Emulator()

# Create the transit AS
transit_as = AutonomousSystem(100)
transit_as.createInternetExchange(1, displayName="IX1")
transit_as.createInternetExchange(2, displayName="IX2")
transit_as.createNetwork("net0")
transit_as.createNetwork("net1")
transit_as.createNetwork("net2")
transit_as.createRouter("router1").joinNetwork("net0")
transit_as.createRouter("router2").joinNetwork("net0").joinNetwork("net1")
transit_as.createRouter("router3").joinNetwork("net1").joinNetwork("net2")
transit_as.createRouter("router4").joinNetwork("net2")

# Create the stub ASes
stub_as1 = AutonomousSystem(200)
stub_as1.createNetwork("net0")
router1 = stub_as1.createRouter("router1").joinNetwork("net0")
stub_as1.createHost("host1").joinNetwork("net0").addService(HostServiceSoftware("CustomSoftware")).addAccount(Account("user", "password"))
stub_as1.createHost("host2").joinNetwork("net0")

stub_as2 = AutonomousSystem(201)
stub_as2.createNetwork("net0")
router2 = stub_as2.createRouter("router2").joinNetwork("net0")
stub_as2.createHost("host1").joinNetwork("net0")
stub_as2.createHost("host2").joinNetwork("net0")

# Utility AS (stub AS3) with custom setup
stub_as3 = makeStubAs(300)
# Customization of stub_as3 goes here

# BGP peering
ebgp = Ebgp()
ebgp.addPeering(transit_as, stub_as1)
ebgp.addPeering(transit_as, stub_as2)
ebgp.addPeering(transit_as, stub_as3)
ebgp.addDirectPeering(stub_as1, stub_as2)

# Web service layer
web_service = WebService()
web_service.addWebService("web1", 80)
web_service.addWebService("web2", 80)
web_service.bind("web1", "200:host1")
web_service.bind("web2", "201:host1")

# Add layers to the emulator
emulator.addLayer(transit_as)
emulator.addLayer(stub_as1)
emulator.addLayer(stub_as2)
emulator.addLayer(stub_as3)
emulator.addLayer(ebgp)
emulator.addLayer(web_service)

# Save to a component file
emulator.saveToFile("emulation_environment.json")

# Render the emulator
emulator.render()

# Change display names for nodes hosting the web services
getNodeByAsnAndName(200, "host1").setDisplayName("Web Server 1")
getNodeByAsnAndName(201, "host1").setDisplayName("Web Server 2")

# Compile the emulator using Docker
emulator.compile(DockerCompilationEngine(), outputFolder="output")

# Specify custom images
_selectImageFor("CustomSoftware", "dockerhub/custom_image:latest")
_selectImageFor("router", "local/router_image:latest")

# Generate Docker files and copy the base container image to the output folder
# This step is assumed to be handled by the compilation engine and specific API calls for these actions are not directly exposed by the seed-emulator library.
```