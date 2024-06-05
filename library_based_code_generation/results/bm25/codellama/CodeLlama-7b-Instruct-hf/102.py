  Here is the Python code that creates an emulation using the seedemu library with the specified parameters:
```
from seedemu import *

# Create an emulation with three layers: Base, Routing, and Ebgp
emulation = createEmulation()
base = emulation.getBase()
routing = emulation.getRouting()
ebgp = emulation.getEbgp()

# Create multiple autonomous systems with their own hosts and routers
as1 = createAutonomousSystem(1, "AS1")
as2 = createAutonomousSystem(2, "AS2")
as3 = createAutonomousSystem(3, "AS3")

# Join the autonomous systems to different networks
as1.joinNetwork("net1")
as2.joinNetwork("net2")
as3.joinNetwork("net3")

# Create a stub autonomous system with a specified ASN and exchange
def createStubAs(asn, exchange):
    as = createAutonomousSystem(asn, "AS" + str(asn))
    router = createRouter(asn, "Router" + str(asn))
    hosts = createHostsOnNetwork(asn, "Hosts" + str(asn))
    as.joinNetwork("net" + str(asn))
    router.joinInternetExchange(exchange)
    return as, router, hosts

# Create a stub autonomous system with ASN 4 and exchange "exchange1"
as4, router4, hosts4 = createStubAs(4, "exchange1")

# Add private peering relationships between different autonomous systems
as1.setAutonomousSystem(as2)
as2.setAutonomousSystem(as1)
as3.setAutonomousSystem(as4)
as4.setAutonomousSystem(as3)

# Add the layers to the emulator and dump the emulator state to a binary file
emulation.addLayer(base)
emulation.addLayer(routing)
emulation.addLayer(ebgp)
emulation.dumpState("emulation.bin")
```
Note that this code assumes that the `seedemu` library is installed and imported as `seedemu`. Additionally, the `createAutonomousSystem`, `createRouter`, `createHostsOnNetwork`, and `joinInternetExchange` functions are defined in the `seedemu` library.