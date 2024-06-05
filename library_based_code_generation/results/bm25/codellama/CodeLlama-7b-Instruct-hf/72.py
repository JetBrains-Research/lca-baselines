  Here is the Python code that creates the emulation described in the instruction:
```
from seed_emulator import *

# Create the emulation base with 5 stub ASes and hosts
emulator_base = makeEmulatorBaseWith5StubASAndHosts()

# Create the three autonomous systems (AS) with AS numbers 150, 151, and 152
as150 = makeStubAsWithHosts(150, ['web'])
as151 = makeStubAsWithHosts(151, ['web'])
as152 = makeStubAsWithHosts(152, ['web'])

# Add the ASes to the emulation base
emulator_base.add_as(as150)
emulator_base.add_as(as151)
emulator_base.add_as(as152)

# Create the network named 'net0' and add it to the ASes
net0 = Network('net0')
as150.add_network(net0)
as151.add_network(net0)
as152.add_network(net0)

# Create the router named 'router0' and add it to the ASes
router0 = Router('router0')
as150.add_router(router0)
as151.add_router(router0)
as152.add_router(router0)

# Create the web service and install it on the 'web' host in each AS
web_service = WebService('web')
as150.add_host(web_service)
as151.add_host(web_service)
as152.add_host(web_service)

# Create the internet exchange with the number 100
ie100 = InternetExchange(100)

# Add the ASes to the internet exchange
ie100.add_as(as150)
ie100.add_as(as151)

# Make AS150 and AS151 peers on the internet exchange
ie100.make_peers(as150, as151)

# Configure AS150 as a provider for AS152
as150.configure_as_evpn_provider(as152)

# Create a cross connection between the routers of AS150 and AS152
as150.router0.add_cross_connection(as152.router0)

# Render and compile the emulation using Docker with self-managed network
emulator_base.render_and_compile(docker_network='self-managed')

# Save the compiled emulation in the directory './cross-connect'
emulator_base.save_emulation('./cross-connect')
```
Note that this code uses the `seed_emulator` library to create the emulation. The `seed_emulator` library provides a set of APIs that can be used to create and configure the emulation. The code above uses these APIs to create the emulation described in the instruction.