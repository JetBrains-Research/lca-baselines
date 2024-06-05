 Here is the Python code that emulates the requirements:

```python
from seedemu import *

def create_stub_as(asn, exchange, num_hosts=1):
    as_obj = ScionAutonomousSystem(asn)
    router = createRouter(as_obj)
    router.setName(f"Router_{asn}")
    hosts = createHostsOnNetwork(num_hosts, router)
    router.joinInternetExchange(exchange)
    return as_obj, router, hosts

def create_emulation():
    emulator = createEmulation()

    base_asns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    exchanges = ["IX1", "IX2", "IX3"]

    for asn in base_asns:
        as_obj, router, hosts = create_stub_as(asn, exchanges[asn % len(exchanges)])
        emulator.createAutonomousSystem(as_obj)

    for i in range(len(base_asns)):
        for j in range(i+1, len(base_asns)):
            as1 = getAutonomousSystem(base_asns[i])
            as2 = getAutonomousSystem(base_asns[j])
            if not doesAContainB(as1, as2) and not doesAContainB(as2, as1):
                router1 = getNodeByAsnAndName(base_asns[i], "Router_" + str(base_asns[i]))
                router2 = getNodeByAsnAndName(base_asns[j], "Router_" + str(base_asns[j]))
                router1.setPrivatePeering(router2)

    emulator.createBase(makeEmulatorBaseWith10StubASAndHosts())
    emulator.createRouting()
    emulator.createEbgp()
    emulator.dump(filename="emulation.bin")

create_emulation()
```

This code creates an emulation with 10 autonomous systems, each with their own hosts and routers. The routers join different networks, and the autonomous systems are connected through internet exchanges. Private peering relationships are also established between different autonomous systems. The layers are added to the emulator and the emulator state is dumped to a binary file named "emulation.bin". The function `create_stub_as` is defined to create a stub autonomous system with a specified ASN and exchange.