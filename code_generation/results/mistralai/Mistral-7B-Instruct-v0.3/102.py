 Here is the Python code that emulates the requirements using the seedemu library:

```python
from seedemu import Emulator, AS, Router, Host, Link, InternetExchange, PrivatePeering

def create_stub_as(asn, exchange):
    as_ = AS(asn)
    router = Router(name=f"AS{asn}-Router")
    host1 = Host(name=f"AS{asn}-Host1")
    host2 = Host(name=f"AS{asn}-Host2")

    as_.add_host(host1)
    as_.add_host(host2)
    as_.add_router(router)

    link1 = Link(router, host1)
    link2 = Link(router, host2)
    as_.add_link(link1)
    as_.add_link(link2)

    router.join_network(as_.network)
    router.join_exchange(exchange)

    return as_

emulator = Emulator()

as1 = create_stub_as(1, "IX1")
as2 = create_stub_as(2, "IX2")
as3 = create_stub_as(3, "IX3")

private_peering1 = PrivatePeering(as1.router, as2.router)
private_peering2 = PrivatePeering(as2.router, as3.router)

emulator.add_layer(as1)
emulator.add_layer(as2)
emulator.add_layer(as3)
emulator.add_layer(InternetExchange("IX1"))
emulator.add_layer(InternetExchange("IX2"))
emulator.add_layer(InternetExchange("IX3"))

emulator.add_link(Link(as1.router, as2.router, private_peering1))
emulator.add_link(Link(as2.router, as3.router, private_peering2))

emulator.dump("emulation.bin")
```

This code creates an emulation with three autonomous systems (AS1, AS2, and AS3) connected through internet exchanges (IX1, IX2, and IX3). It also defines a function `create_stub_as` to create a stub autonomous system with a specified ASN and exchange, and adds private peering relationships between different autonomous systems. Finally, it adds the layers to the emulator and dumps the emulator state to a binary file named "emulation.bin".