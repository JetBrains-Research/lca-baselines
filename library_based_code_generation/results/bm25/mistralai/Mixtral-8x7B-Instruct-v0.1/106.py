 ```
from seed-emulator import PoAServer, createEmulation, AutonomousSystem, InternetExchange, BgpAttackerComponent

def create\_stub\_as(name, asn, ix\_name):
ix = getInternetExchange(ix\_name)
asn\_obj = createAutonomousSystem(asn)
web\_server = asn\_obj.addComponent("webserver", "webserver")
router = asn\_obj.addComponent("router", "router")
asn\_obj.addLink(web_server, router)
asn\_obj.joinNetwork(ix)
return asn\_obj

ix1 = createInternetExchange("ix1")
ix2 = createInternetExchange("ix2")
ix3 = createInternetExchange("ix3")

stub\_asns = [create\_stub\_as(f"stub{i}", i, "ix1") for i in range(1, 4)]
stub\_asns += [create\_stub_as(f"stub{i}", i, "ix2") for i in range(4, 7)]
stub\_asns += [create\_stub_as(f"stub{i}", i, "ix3") for i in range(7, 10)]

asn1 = createAutonomousSystem(101)
asn1\_router1 = asn1.addComponent("router1", "router")
asn1\_router2 = asn1.addComponent("router2", "router")
asn1.addLink(asn1\_router1, asn1\_router2)
asn1.joinNetwork(getInternetExchange("ix1"))
asn1.joinNetwork(getInternetExchange("ix2"))

asn2 = createAutonomousSystem(201)
asn2\_router1 = asn2.addComponent("router1", "router")
asn2\_router2 = asn2.addComponent("router2", "router")
asn2.addLink(asn2\_router1, asn2\_router2)
asn2.joinNetwork(getInternetExchange("ix1"))
asn2.joinNetwork(getInternetExchange("ix3"))

for asn1\_router in [asn1\_router1, asn1\_router2]:
for asn2\_router in [asn2\_router1, asn2\_router2]:
asn1\_router.addPeer(asn2\_router)

bgp\_attacker = BgpAttackerComponent("attacker")
bgp\_attacker.hijackPrefix("1.2.3.0/24")
bgp\_attacker.hijackPrefix("5.6.7.0/24")
ix1 = getInternetExchange("ix1")
setInternetExchange(ix1.name, ix1.addComponent(bgp_attacker))

emulation = createEmulation(
merged=[
PoAServer.shouldMerge(lambda a, b: doesAContainB(a.autonomousSystems, b.autonomousSystems)),
BgpAttackerComponent.shouldMerge
],
autonomous\_systems=[asn1, asn2] + stub\_asns,
internet\_exchanges=[ix1, ix2, ix3]
)

emulation.compile("output\_directory")
```