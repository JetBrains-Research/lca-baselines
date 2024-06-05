 Here is a simplified Python code snippet that uses the seed-emulator library to create the emulation as per your instructions. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, ensure you have the seed-emulator library installed.

```python
from seed_emulator import PoAServer, InternetExchange, AutonomousSystem, BgpAttackerComponent

def create_emulation():
    poa_server = PoAServer()

    # Create internet exchanges
    ix1 = create_internet_exchange(poa_server, "ix1")
    ix2 = create_internet_exchange(poa_server, "ix2")
    ix3 = create_internet_exchange(poa_server, "ix3")

    # Create stub autonomous systems with web servers and routers
    stub_as1 = make_stub_as(poa_server, "stub_as1", ix1)
    stub_as2 = make_stub_as(poa_server, "stub_as2", ix2)
    stub_as3 = make_stub_as(poa_server, "stub_as3", ix3)

    # Create autonomous systems with routers joining different networks and internet exchanges
    asn4 = create_autonomous_system(poa_server, "asn4")
    asn5 = create_autonomous_system(poa_server, "asn5")
    join_network(asn4, "network1", ix1)
    join_network(asn5, "network2", ix3)

    # Private peerings between different autonomous systems
    set_private_peering(asn4, asn5, ix1)

    # BGP attacker component
    attacker = BgpAttackerComponent()
    attacker.hijack_prefixes(["1.0.0.0/8", "2.0.0.0/8"])
    attacker.join_internet_exchange(ix1)

    # Merge the BGP attacker with the emulator
    emulator = poa_server.shouldMerge(attacker)

    # Render and compile the new emulator using Docker
    emulator.render("emulator", "docker")
    emulator.compile("emulator", "docker", "output_directory")

def create_internet_exchange(poa_server, name):
    return poa_server.createInternetExchange(name)

def make_stub_as(poa_server, asn, ix):
    web_server = poa_server.makeWebServer()
    router = poa_server.makeRouter()
    router.joinNetwork(ix)
    router.setAutonomousSystem(AutonomousSystem(asn))
    router.addHost(web_server)
    return router

def create_autonomous_system(poa_server, asn):
    return AutonomousSystem(asn)

def join_network(asn, network, ix):
    router = get_router_by_asn_and_name(asn, "router1")
    router.joinNetwork(ix)
    router.setAutonomousSystem(asn)

def get_router_by_asn_and_name(asn, name):
    return poa_server.getNodeByAsnAndName(asn, name)

def set_private_peering(asn1, asn2, ix):
    router1 = get_router_by_asn_and_name(asn1, "router1")
    router2 = get_router_by_asn_and_name(asn2, "router1")
    router1.setInternetExchange(ix)
    router2.setInternetExchange(ix)
```

This code creates an emulation with the specified layers, creates multiple stub autonomous systems, defines private peerings, and adds a BGP attacker component. The emulation is then merged, rendered, and compiled using Docker. The resulting emulator is saved in the "output_directory".