 Here is a Python code snippet that creates an emulation environment using the seedemu library as per your instructions:

```python
from seedemu import Emulator, PoAServer, ScionAutonomousSystem, AutonomousSystem, makeEmulatorBaseWith10StubASAndHosts, makeStubAsWithHosts, getNodeByAsnAndName, addPrivatePeering, up_emulator, toGraphviz, to_json, resolveTo, get_balance_with_name, createEmulation

# Create Internet Exchanges
ix1 = AutonomousSystem(name="IX-1", asn=64512)
ix2 = AutonomousSystem(name="IX-2", asn=64513)

# Create Transit Autonomous Systems
transit_as1 = ScionAutonomousSystem(name="Transit-AS1", asn=64514)
transit_as2 = ScionAutonomousSystem(name="Transit-AS2", asn=64515)

# Create single-homed stub Autonomous Systems
stub_as1 = ScionAutonomousSystem(name="Stub-AS1", asn=64516)
stub_as2 = ScionAutonomousSystem(name="Stub-AS2", asn=64517)

# Create a real-world Autonomous System
real_world_as = ScionAutonomousSystem(name="Real-World-AS", asn=64518)

# Add hosts to the Autonomous Systems
base = makeEmulatorBaseWith10StubASAndHosts()
for asn, as_obj in base.items():
    for host in as_obj.hosts:
        as_obj.hosts.remove(host)

    # Add custom host to Stub-AS1
    custom_host = makeStubAsWithHosts(asn=stub_as1.asn, host_count=1)
    custom_host[0].ip = "192.168.1.1"
    stub_as1.hosts += custom_host[0].hosts

    # Add the Autonomous Systems to the base
    base[asn] = as_obj

# Add Internet Exchanges to the base
base[ix1.asn].peers += [ix1]
base[ix2.asn].peers += [ix2]

# Add Transit Autonomous Systems to the base
base[transit_as1.asn].peers += [transit_as2]

# Add private peering between Stub-AS1 and Transit-AS1
addPrivatePeering(stub_as1, transit_as1)

# Add real-world Autonomous System to the base
base[real_world_as.asn] = real_world_as

# Enable remote access to Stub-AS1's network
remote_access = PoAServer(name="Remote-Access", asn=64519)
remote_access.peers += [stub_as1]
base[64519] = remote_access

# Set up peering via a route server
route_server = AutonomousSystem(name="Route-Server", asn=64520)
route_server_peers = [transit_as1, transit_as2, real_world_as]
for peer in route_server_peers:
    addPrivatePeering(peer, route_server)

# Add all the layers to the emulator
emulator = createEmulation(base)

# Save the emulator to a component file
emulator.save("emulator.component")

# Render and compile the emulator
up_emulator(emulator)
emulator.toGraphviz("emulator.dot")
emulator.to_json("emulator.json")
```

Please note that this code assumes that you have the seedemu library installed. If you haven't, you can install it using pip:

```
pip install seedemu
```