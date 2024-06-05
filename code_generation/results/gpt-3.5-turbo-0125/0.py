import seedemu

# Create emulator
emulator = seedemu.Emulator()

# Create base layer
base_layer = seedemu.BaseLayer()
autonomous_system1 = base_layer.create_autonomous_system()
autonomous_system2 = base_layer.create_autonomous_system()
internet_exchange1 = base_layer.create_internet_exchange()
internet_exchange2 = base_layer.create_internet_exchange()

for i in range(5):
    host = autonomous_system1.create_host()
    router = autonomous_system1.create_router()
    autonomous_system1.add_host_to_network(host)
    autonomous_system1.add_router_to_network(router)
    router.join_internet_exchange(internet_exchange1)

for i in range(3):
    host = autonomous_system2.create_host()
    router = autonomous_system2.create_router()
    autonomous_system2.add_host_to_network(host)
    autonomous_system2.add_router_to_network(router)
    router.join_internet_exchange(internet_exchange2)

# Create domain name caching service
domain_name_caching_service = seedemu.DomainNameCachingService()
host1 = autonomous_system1.get_hosts()[0]
host2 = autonomous_system2.get_hosts()[0]
domain_name_caching_service.install_on_host(host1)
domain_name_caching_service.install_on_host(host2)

# Add bindings for domain name caching service
domain_name_caching_service.add_binding(host1, "example.com", "192.168.1.1")
domain_name_caching_service.add_binding(host2, "example.net", "192.168.2.1")

# Create eBGP layer
ebgp_layer = seedemu.EBGPLayer()
ebgp_layer.add_private_peering(autonomous_system1.get_router(), autonomous_system2.get_router())

# Add layers to emulator
emulator.add_layer(base_layer)
emulator.add_layer(ebgp_layer)
emulator.add_service(domain_name_caching_service)

# Dump state of emulator to binary file
emulator.dump_state("emulator_state.bin")