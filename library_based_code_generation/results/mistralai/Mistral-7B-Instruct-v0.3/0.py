 Here is a simplified example of how you might set up the emulation using the `seedemu` library. Note that this is a basic example and may not cover all your requirements. You may need to adjust it according to your specific needs.

```python
from seedemu import Emulator, AutonomousSystem, Router, Host, InternetExchange, eBGP, DomainNameCachingService

# Create the emulator
emulator = Emulator()

# Create autonomous systems
as1 = AutonomousSystem(as_number=64512)
as2 = AutonomousSystem(as_number=64513)

# Add internet exchanges
ixp = InternetExchange()

# Add autonomous systems and internet exchange to the emulator
emulator.add(as1)
emulator.add(as2)
emulator.add(ixp)

# Add routers to the autonomous systems
router1 = Router(name='router1')
router2 = Router(name='router2')
as1.add_router(router1)
as2.add_router(router2)

# Add hosts to the autonomous systems
host1 = Host(name='host1')
host2 = Host(name='host2')
router1.add_host(host1)
router1.add_host(host2)

# Add hosts to the internet exchange
ixp.add_router(router1)
ixp.add_router(router2)

# Configure eBGP between routers
eBGP(router1, router2, as_number=as1.as_number, peer_as=as2.as_number)
eBGP(router2, router1, as_number=as2.as_number, peer_as=as1.as_number)

# Install domain name caching service on specific hosts
dncs = [DomainNameCachingService(host1), DomainNameCachingService(host2)]

# Add domain name caching services to the emulator
for dnc in dncs:
    emulator.add(dnc)

# Add bindings for the domain name caching services
for dnc in dncs:
    dnc.add_binding('example.com', '10.0.0.1')  # replace with your IP

# Add all layers to the emulator
emulator.add(as1)
emulator.add(as2)
emulator.add(ixp)
for dnc in dncs:
    emulator.add(dnc)

# Dump the state of the emulator to a binary file
emulator.dump('emulation.bin')
```

This code creates an emulation with two autonomous systems, two routers, four hosts, an internet exchange, and a domain name caching service. The routers are configured to peer with each other using eBGP, and the domain name caching service is installed on two specific hosts with a binding for 'example.com'. The state of the emulator is then dumped to a binary file.