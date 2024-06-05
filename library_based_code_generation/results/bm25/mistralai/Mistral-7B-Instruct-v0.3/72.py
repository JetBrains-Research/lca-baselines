 Here is the Python code that emulates the network as per your instructions, using the seed-emulator library:

```python
import seed_emulator as se

# Create the emulation base with 5 ASes and hosts
emulation_base = se.makeEmulatorBaseWith5StubASAndHosts()

# Create AS150, AS151, and AS152
as150 = se.makeStubAsWithHosts(150, ['web', 'router0'])
as151 = se.makeStubAsWithHosts(151, ['web', 'router0'])
as152 = se.makeStubAsWithHosts(152, ['web', 'router0'])

# Add the ASes to the emulation base
emulation_base.addAS(as150)
emulation_base.addAS(as151)
emulation_base.addAS(as152)

# Configure networks for each AS
as150_net0 = se.Network('net0', as150)
as151_net0 = se.Network('net0', as151)
as152_net0 = se.Network('net0', as152)

as150_net0.addHost(as150.getHostByName('web'))
as150_net0.addHost(as150.getHostByName('router0'))
as151_net0.addHost(as151.getHostByName('web'))
as151_net0.addHost(as151.getHostByName('router0'))
as152_net0.addHost(as152.getHostByName('web'))
as152_net0.addHost(as152.getHostByName('router0'))

emulation_base.addNetwork(as150_net0)
emulation_base.addNetwork(as151_net0)
emulation_base.addNetwork(as152_net0)

# Connect AS150 and AS152 routers
as150_router0 = as150.getHostByName('router0')
as152_router0 = as152.getHostByName('router0')
as150_router0.addLink(as152_router0)

# Create internet exchange with AS number 100
ix = se.PoAServer(100)

# Peer AS150 and AS151 on the internet exchange
as150_ix_peer = se.WebServer(as150, ix, se.DefaultWebServiceMerger())
as151_ix_peer = se.WebServer(as151, ix, se.DefaultWebServiceMerger())
ix.addPeer(as150_ix_peer)
ix.addPeer(as151_ix_peer)

# Make AS150 a provider for AS152
as150_provider = se.makeTransitAs(as150)
as152_customer = se.makeStubAsWithHosts(152, ['web'], transit_as=as150.getAsn())
as152_customer.markAsEdge()
as150_provider.addCustomer(as152_customer)

# Create web services for each 'web' host
for as_ in [as150, as151, as152]:
    web_host = as_.getHostByName('web')
    web_service = se.WebService(web_host)
    web_host.addService(web_service)

# Render and compile the emulation using Docker with self-managed network
emulation = se.createEmulation(emulation_base, self_managed_network=True)
emulation.renderAndCompile(docker_directory='./cross-connect')
```

This code creates the emulation as per your instructions, using the seed-emulator library. The compiled emulation will be saved in the directory './cross-connect'.