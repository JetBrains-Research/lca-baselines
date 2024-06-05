import seed_emulator

emulation = seed_emulator.Emulation()

as150 = emulation.create_autonomous_system(150)
as151 = emulation.create_autonomous_system(151)
as152 = emulation.create_autonomous_system(152)

web_host = emulation.create_host('web')
router0 = emulation.create_router('router0')

as150.add_host(web_host)
as150.add_router(router0)
as151.add_host(web_host)
as151.add_router(router0)
as152.add_host(web_host)
as152.add_router(router0)

net0 = emulation.create_network('net0')
net0.add_node(web_host)
net0.add_node(router0)

as150.connect_to(as152, 'router0', 'router0')

internet_exchange = emulation.create_internet_exchange(100)
internet_exchange.add_peer(as150)
internet_exchange.add_peer(as151)

as150.add_provider(as152)

emulation.render_with_docker(network_mode='self-managed', output_directory='./cross-connect')