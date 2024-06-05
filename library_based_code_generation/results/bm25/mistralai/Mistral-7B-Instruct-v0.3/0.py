 Here is a Python code snippet that uses the seedemu library to create the emulation as per your instructions. Please note that this is a simplified version and you might need to adjust it according to your specific requirements.

```python
from seedemu.api import shouldMerge, makeEmulatorBaseWith5StubASAndHosts, DomainNameCachingService, AutonomousSystem, getNodeByAsnAndName, createAutonomousSystem, DomainNameCachingServer, DefaultDomainNameCachingServiceMerger, ScionAutonomousSystem, __configureAutonomousSystem, setAutonomousSystem, getAutonomousSystem, DomainNameService, ReverseDomainNameService, createHostsOnNetwork, DefaultDomainNameServiceMerger, joinInternetExchange, PoAServer, DomainRegistrarService

# Create the emulator with 5 autonomous systems and hosts
emulator = makeEmulatorBaseWith5StubASAndHosts()

# Configure autonomous systems, internet exchanges, and hosts
for asn in range(1, 6):
    as_ = createAutonomousSystem(asn)
    __configureAutonomousSystem(as_, asn)
    setAutonomousSystem(as_, f"AS{asn}")

    # Add hosts and routers to the autonomous system
    hosts = getAutonomousSystem(as_).getHosts()
    routers = [host for host in hosts if host.getType() == "router"]
    network = getAutonomousSystem(as_).getNetwork()
    createHostsOnNetwork(network, len(routers) - 1)

    # Join the internet exchange
    ie = PoAServer(f"ie-{asn}")
    routers[0].joinInternetExchange(ie)

# Install domain name caching service on specific hosts
dns_servers = [hosts[i] for i in [1, 3] if i % 2 == 0]
for host in dns_servers:
    dns = DomainNameCachingService(DomainNameCachingServer)
    shouldMerge(dns, DefaultDomainNameCachingServiceMerger)
    host.addService(dns)

# Add bindings for the domain name caching service installations
for host in dns_servers:
    dns = host.getServiceByName("DomainNameCachingService")
    dns.addBinding("example.com", "127.0.0.1")
    dns.addBinding("example.net", "127.0.0.1")

# Create eBGP layer and add private peering between autonomous systems
for asn1, asn2 in zip(range(1, 6), range(2, 6)):
    as1 = getAutonomousSystem(asn1)
    as2 = getAutonomousSystem(asn2)
    bgp = ScionAutonomousSystem(as1.getAsn(), as2.getAsn())
    as1_router = getNodeByAsnAndName(emulator, as1.getAsn(), routers[0].getName())
    as2_router = getNodeByAsnAndName(emulator, as2.getAsn(), routers[1].getName())
    bgp.addRouter(as1_router)
    bgp.addRouter(as2_router)
    as1_router.addService(bgp)
    as2_router.addService(bgp)

# Add domain name service and reverse domain name service
dns = DomainNameService()
shouldMerge(dns, DefaultDomainNameServiceMerger)
emulator.addService(dns)
rdns = ReverseDomainNameService()
shouldMerge(rdns, DefaultDomainNameServiceMerger)
emulator.addService(rdns)

# Add domain registrar service
drs = DomainRegistrarService()
shouldMerge(drs, DefaultDomainNameServiceMerger)
emulator.addService(drs)

# Dump the state of the emulator to a binary file
emulator.dumpState("emulation.bin")
```

This code creates an emulation with 5 autonomous systems, each with multiple hosts and a router. The hosts and routers join networks within their autonomous systems and internet exchanges. A domain name caching service is installed on specific hosts, and bindings are added for these installations. The eBGP layer is added with private peering between different autonomous systems. Finally, all the layers and the domain name caching service are added to the emulator, and the state of the emulator is dumped to a binary file.