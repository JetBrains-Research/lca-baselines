 ```python
from seedemu.api import *

# Create emulation base with 10 stub ASes and hosts
base = makeEmulatorBaseWith10StubASAndHosts()

# Create custom display names for Internet Exchanges
ix_display_names = {
    1: "IX1",
    2: "IX2",
    3: "IX3"
}

# Create Internet Exchanges with custom display names
for asn, name in ix_display_names.items():
    base.addInternetExchange(asn, name)

# Create transit and single-homed stub ASes with various services
transit_asns = [100, 200]
stub_asns = [300, 301, 302, 303, 304]
for asn in transit_asns:
    base.createAutonomousSystem(asn, PoAServer)
for asn in stub_asns:
    base.createAutonomousSystem(asn, ScionAutonomousSystem)

# Add a host with a customized IP address to one of the Autonomous Systems
host_asn = 300
host_ip = "10.0.0.1"
base.setAutonomousSystem(host_asn, __configureAutonomousSystem(
    base.getAutonomousSystem(host_asn),
    makeStubAsWithHosts(host_ip)
))

# Create a real-world Autonomous System
real_world_asn = 400
base.createAutonomousSystem(real_world_asn, PoAServer)

# Enable remote access to one of the Autonomous System's network
remote_asn = 200
base.addRemoteAccess(remote_asn, "0.0.0.0/0")

# Set up peering via a route server and private peering with different peer relationships
route_server_asn = 100
for asn in [200, 300, 400]:
    if asn != route_server_asn:
        base.addPrivatePeering(route_server_asn, asn)

# Add layers to the emulator
base.addLayer(Ebgp)
base.addLayer(Ibgp)
base.addLayer(Ospf)
base.addLayer(WebService)

# Save the emulator to a component file
base.toGraphviz("emulator.dot")

# Render and compile the emulator
up_emulator("emulator.dot")
```