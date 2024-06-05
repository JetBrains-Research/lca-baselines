import seed_emulator

emulator = seed_emulator.Emulator()

AS150 = seed_emulator.AutonomousSystem("AS150")
AS151 = seed_emulator.AutonomousSystem("AS151")
AS152 = seed_emulator.AutonomousSystem("AS152")

for i in range(4):
    router = seed_emulator.Router(f"Router{i}", AS150)
    AS150.add_router(router)

for i in range(3):
    network = seed_emulator.Network(f"Network{i}")
    AS150.add_network(network)

web_host151 = seed_emulator.WebHost("Web Host AS151", AS151)
router151 = seed_emulator.Router("Router AS151", AS151)
network151 = seed_emulator.Network("Network AS151")
AS151.add_host(web_host151)
AS151.add_router(router151)
AS151.add_network(network151)

web_host152 = seed_emulator.WebHost("Web Host AS152", AS152)
router152 = seed_emulator.Router("Router AS152", AS152)
network152 = seed_emulator.Network("Network AS152")
AS152.add_host(web_host152)
AS152.add_router(router152)
AS152.add_network(network152)

AS151.join_internet_exchange()
AS152.join_internet_exchange()

AS150.add_bgp_peering(AS151)
AS150.add_bgp_peering(AS152)

emulator.add_autonomous_system(AS150)
emulator.add_autonomous_system(AS151)
emulator.add_autonomous_system(AS152)

emulator.dump_state("emulator_state.bin")