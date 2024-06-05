



 Here is the Python code that meets the requirements:

```python
from seed_emulator import *

# Create the emulation
emulation = Emulation()

# Create the ASes
as150 = makeStubAsWithHosts(150, 1, 1)
as151 = makeStubAsWithHosts(151, 1, 1)
as152 = makeStubAsWithHosts(152, 1, 1)

# Add the ASes to the emulation
emulation.addAs(as150)
emulation.addAs(as151)
emulation.addAs(as152)

# Create the network
net0 = Network('net0')

# Add the network to the ASes
as150.addNetwork(net0)
as151.addNetwork(net0)
as152.addNetwork(net0)

# Add the web service to the 'web' hosts
as150.getHost('web').addService(WebService())
as151.getHost('web').addService(WebService())
as152.getHost('web').addService(WebService())

# Create the internet exchange
ix100 = InternetExchange(100)

# Add the internet exchange to the emulation
emulation.addIx(ix100)

# Add AS150 and AS151 as peers on the internet exchange
ix100.addPeer(as150)
ix150.addPeer(as151)

# Add AS152 as a provider for AS150
as150.addProvider(as152)

# Create the cross connection between AS150 and AS152
as150.getRouter('router0').addPeer(as152.getRouter('router0'))

# Render the emulation
emulation.render()

# Compile the emulation using Docker with self-managed network
emulation.compile(docker=True, self_managed_network=True)

# Save the compiled emulation in the './cross-connect' directory
emulation.save('./cross-connect')
```

This code creates an emulation using the seed-emulator library. It includes three autonomous systems (AS) with AS numbers 150, 151, and 152. Each AS has a host named 'web' and a router named 'router0'. The 'web' host in each AS has a web service installed. Each AS also has a network named 'net0' which both the 'web' host and 'router0' join. AS150 and AS152 have a cross connection between their routers. An internet exchange with the number 100 is created, and AS150 and AS151 are peers on this exchange. AS150 is also a provider for AS152. The emulation is rendered and compiled using Docker with self-managed network. The compiled emulation is saved in the directory './cross-connect'.

Please note that this code is not tested and may require some modifications to work properly. Source: assistant
EOT: true