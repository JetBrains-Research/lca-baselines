  Here is the Python code that creates an emulation using the seed-emulator library based on the given instructions:
```
from seed_emulator import *

# Create an emulator with three autonomous systems
emulator = makeEmulatorBaseWith5StubASAndHosts(3)

# Add the transit AS with four routers and three networks
transit_as = getNodeByAsnAndName(emulator, 150, "AS150")
transit_as.addRouter("r1", "10.0.0.1")
transit_as.addRouter("r2", "10.0.0.2")
transit_as.addRouter("r3", "10.0.0.3")
transit_as.addRouter("r4", "10.0.0.4")
transit_as.addNetwork("n1", "10.0.0.0/24")
transit_as.addNetwork("n2", "10.0.1.0/24")
transit_as.addNetwork("n3", "10.0.2.0/24")

# Add the web hosts and routers for AS151 and AS152
as151 = getNodeByAsnAndName(emulator, 151, "AS151")
as151.addRouter("r5", "10.0.3.1")
as151.addHost("h1", "10.0.3.2")
as152 = getNodeByAsnAndName(emulator, 152, "AS152")
as152.addRouter("r6", "10.0.4.1")
as152.addHost("h2", "10.0.4.2")

# Join the networks and internet exchange
joinNetwork(transit_as, "n1")
joinNetwork(transit_as, "n2")
joinNetwork(transit_as, "n3")
joinInternetExchange(transit_as, "n1")
joinInternetExchange(transit_as, "n2")
joinInternetExchange(transit_as, "n3")

# Set up BGP peering between AS150 and AS151
setInternetExchange(transit_as, "n1", "AS151")
setInternetExchange(transit_as, "n2", "AS151")
setInternetExchange(transit_as, "n3", "AS151")

# Set up BGP peering between AS150 and AS152
setInternetExchange(transit_as, "n1", "AS152")
setInternetExchange(transit_as, "n2", "AS152")
setInternetExchange(transit_as, "n3", "AS152")

# Add all the layers to the emulator
emulator.addLayer(transit_as)
emulator.addLayer(as151)
emulator.addLayer(as152)

# Dump the emulator's state to a binary file
gen_emulation_files(emulator, "emulation.bin")
```
Note that this code assumes that the seed-emulator library is installed and imported as `seed_emulator`.