import seed_emulator

emulator = seed_emulator.Emulator()

# Base layer
base_layer = seed_emulator.Layer("base")
as1 = base_layer.create_autonomous_system()
as1.create_router("router1")
for i in range(5):
    as1.create_host(f"host{i+1}")
as1.create_network("network1")

as2 = base_layer.create_autonomous_system()
for i in range(3):
    as2.create_router(f"router{i+1}")
    as2.create_network(f"network{i+1}")

as3 = base_layer.create_autonomous_system()
as3.create_router("router1")
as3.create_router("router2")
as3.create_network("network1")

emulator.add_layer(base_layer)

# Routing layer
routing_layer = seed_emulator.Layer("routing")
# Add routing configurations here

emulator.add_layer(routing_layer)

# eBGP layer
ebgp_layer = seed_emulator.Layer("eBGP")
# Add eBGP configurations here

emulator.add_layer(ebgp_layer)

# Dump emulator state to binary file
emulator.dump_state("emulator_state.bin")