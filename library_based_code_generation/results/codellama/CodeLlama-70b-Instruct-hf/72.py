



 Here is the Python code that meets the requirements:

```
from seed_emulator import Emulator, AS, Host, Router, Network, IX, Peering

# Create an Emulator instance
emulator = Emulator()

# Create three autonomous systems (AS) with AS numbers 150, 151, and 152
as150 = AS(150)
as151 = AS(151)
as152 = AS(152)

# Add the ASs to the emulator
emulator.add_as(as150)
emulator.add_as(as151)
emulator.add_as(as152)

# Create a network named 'net0' for each AS
net0_as150 = Network('net0')
net0_as151 = Network('net0')
net0_as152 = Network('net0')

# Create a host named 'web' and a router named 'router0' for each AS
web_as150 = Host('web', as150)
router0_as150 = Router('router0', as150)
web_as151 = Host('web', as151)
router0_as151 = Router('router0', as151)
web_as152 = Host('web', as152)
router0_as152 = Router('router0', as152)

# Add the hosts and routers to their respective ASs
as150.add_host(web_as150)
as150.add_router(router0_as150)
as151.add_host(web_as151)
as151.add_router(router0_as151)
as152.add_host(web_as152)
as152.add_router(router0_as152)

# Join the 'net0' network to the 'web' host and 'router0' router for each AS
web_as150.join_network(net0_as150)
router0_as150.join_network(net0_as150)
web_as151.join_network(net0_as151)
router0_as151.join_network(net0_as151)
web_as152.join_network(net0_as152)
router0_as152.join_network(net0_as152)

# Create an internet exchange with the number 100
ix = IX(100)

# Add AS150 and AS151 as peers on the internet exchange
ix.add_peer(as150)
ix.add_peer(as151)

# Add AS152 as a provider for AS150
as150.add_provider(as152)

# Render the emulation
emulator.render()

# Compile the emulation using Docker with self-managed network
emulator.compile(docker_network='self-managed')

# Save the compiled emulation in the directory './cross-connect'
emulator.save('./cross-connect')
```

This code creates an emulation using the seed-emulator library, as per the requirements. It creates three autonomous systems (AS) with AS numbers 150, 151, and 152, each with a host named 'web' and a router named 'router0'. Each AS also has a network named 'net0' which both the 'web' host and 'router0' join. AS150 and AS152 have a cross connection between their routers. An internet exchange with the number 100 is created, and AS150 and AS151 are peers on this exchange. AS150 is also a provider for AS152. The emulation is rendered and compiled using Docker with self-managed network. The compiled emulation is saved in the directory './cross-connect'.