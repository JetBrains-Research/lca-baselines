 Here is the Python code that emulates the given requirements using the `seed-emulator` library:

```python
from seed import Emulator, Network, AutonomousSystem, Router, Host, Link, Bgp

# Create the emulator
emulator = Emulator()

# Base layer
as1 = AutonomousSystem(name='AS1')
for i in range(1, 6):
    if i == 5:
        r1 = Router(name=f'r{i}')
    else:
        r1 = Router(name=f'r{i}')
        h1 = Host(name=f'h{i}')
    r1.add_interface(Network('10.0.0.0/24'))
    r1.set_ip('10.0.0.1', '10.0.0.255')
    r1.add_interface(Network(f'10.1.{i}.0/24'))
    r1.set_ip(f'10.1.{i}.1', f'10.1.{i}.255')
    if i != 5:
        link = Link(endpoints=[r1, r5])
        emulator.add_link(link)
    as1.add_router(r1)
    if i != 5:
        as1.add_host(h1)

as2 = AutonomousSystem(name='AS2')
for i in range(1, 4):
    r2i = Router(name=f'r2{i}')
    r2i.add_interface(Network(f'10.2.{i}.0/24'))
    r2i.set_ip(f'10.2.{i}.1', f'10.2.{i}.255')
    as2.add_router(r2i)

as3 = AutonomousSystem(name='AS3')
r31 = Router(name='r31')
r31.add_interface(Network('10.3.0.0/24'))
r31.set_ip('10.3.0.1', '10.3.0.255')
r32 = Router(name='r32')
r32.add_interface(Network('10.3.0.0/24'))
r32.set_ip('10.3.0.2', '10.3.0.255')
as3.add_router(r31)
as3.add_router(r32)

# Routing layer
as1_as2_link = Link(endpoints=[r5, r21])
as1_as3_link = Link(endpoints=[r5, r31])
emulator.add_link(as1_as2_link)
emulator.add_link(as1_as3_link)

# eBGP layer
bgp_as1 = Bgp(as_number=65001, router=r5)
bgp_as2 = Bgp(as_number=65002, router=r21)
bgp_as3 = Bgp(as_number=65003, router=r31)

bgp_as1.add_peer(bgp_as2, '10.0.0.2', '65002')
bgp_as1.add_peer(bgp_as3, '10.0.0.3', '65003')
bgp_as2.add_peer(bgp_as1, '10.2.1.2', '65001')
bgp_as3.add_peer(bgp_as1, '10.3.0.3', '65001')

# Add all layers to the emulator
emulator.add_autonomous_system(as1)
emulator.add_autonomous_system(as2)
emulator.add_autonomous_system(as3)

# Dump the emulator state to a binary file
emulator.dump('emulation.bin')
```

This code creates an emulation with three autonomous systems (AS1, AS2, and AS3) as specified. The base layer sets up the required routers and hosts, the routing layer adds links between different autonomous systems, and the eBGP layer establishes private peering between them. Finally, the emulator state is dumped to a binary file named 'emulation.bin'.