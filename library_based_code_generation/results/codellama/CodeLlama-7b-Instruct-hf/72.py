  from seed_emulator import Emulation

emulation = Emulation()

# Create three autonomous systems
as150 = emulation.create_as(150)
as151 = emulation.create_as(151)
as152 = emulation.create_as(152)

# Create a host named 'web' in each AS
web150 = as150.create_host('web')
web151 = as151.create_host('web')
web152 = as152.create_host('web')

# Create a router named 'router0' in each AS
router0150 = as150.create_router('router0')
router0151 = as151.create_router('router0')
router0152 = as152.create_router('router0')

# Create a network named 'net0' in each AS
net0150 = as150.create_network('net0')
net0151 = as151.create_network('net0')
net0152 = as152.create_network('net0')

# Connect the 'web' host and 'router0' to the 'net0' network
web150.connect_to(net0150)
router0150.connect_to(net0150)
web151.connect_to(net0151)
router0151.connect_to(net0151)
web152.connect_to(net0152)
router0152.connect_to(net0152)

# Create an internet exchange with the number 100
ie100 = emulation.create_ie(100)

# Make AS150 and AS151 peers on the internet exchange
ie100.add_peer(as150)
ie100.add_peer(as151)

# Make AS150 a provider for AS152
as150.add_provider(as152)

# Create a cross connection between the routers of AS150 and AS152
router0150.create_cross_connection(router0152)

# Render and compile the emulation using Docker with self-managed network
emulation.render_and_compile(docker_network='self-managed')

# Save the compiled emulation to the directory './cross-connect'
emulation.save('./cross-connect')