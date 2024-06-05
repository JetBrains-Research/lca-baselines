 ```
from seedemu.api import shouldMerge, createAutonomousSystem, AutonomousSystem, PoAServer, makeEmulatorBaseWith5StubASAndHosts, makeEmulatorBaseWith10StubASAndHosts, ScionAutonomousSystem, getAutonomousSystem, __configureAutonomousSystem, setAutonomousSystem, doesAContainB, joinInternetExchange, getNodeByAsnAndName, createEmulation, joinNetwork, createInternetExchange, createRouter, createHostsOnNetwork, makeStubAsWithHosts

def create_stub_as(asn, exchange):
as = createAutonomousSystem(asn, ScionAutonomousSystem)
__configureAutonomousSystem(as)

hosts = createHostsOnNetwork(as, "host%d" % (1,), 1)
router = createRouter(as, "r%d" % (1,))

joinNetwork(router, hosts[0].network)
joinInternetExchange(router, exchange)

return as

base = makeEmulatorBaseWith10StubASAndHosts()

exchange1 = createInternetExchange("ix1")
exchange2 = createInternetExchange("ix2")

as1 = create_stub_as(100, exchange1)
as2 = create_stub_as(200, exchange2)
as3 = create_stub_as(300, exchange1)

joinInternetExchange(getNodeByAsnAndName(as2, "r1"), exchange1)
joinInternetExchange(getNodeByAsnAndName(as3, "r1"), exchange2)

createEmulation("emulation.bin", base)
```