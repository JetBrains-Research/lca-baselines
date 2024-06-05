  from seed_emulator import Emulator

emulator = Emulator()

# AS150: Transit AS with four routers and three networks
as150 = emulator.add_as("AS150")
as150.add_router("r1")
as150.add_router("r2")
as150.add_router("r3")
as150.add_router("r4")
as150.add_network("n1")
as150.add_network("n2")
as150.add_network("n3")

# AS151: Web host and router
as151 = emulator.add_as("AS151")
as151.add_router("r5")
as151.add_web_host("w1")

# AS152: Web host and router
as152 = emulator.add_as("AS152")
as152.add_router("r6")
as152.add_web_host("w2")

# Internet exchange
ix = emulator.add_ix("ix")

# BGP peering between AS150 and AS151
as150.add_bgp_peer(as151, "r1")
as150.add_bgp_peer(as151, "r2")
as150.add_bgp_peer(as151, "r3")
as150.add_bgp_peer(as151, "r4")

# BGP peering between AS150 and AS152
as150.add_bgp_peer(as152, "r1")
as150.add_bgp_peer(as152, "r2")
as150.add_bgp_peer(as152, "r3")
as150.add_bgp_peer(as152, "r4")

# Join networks
as151.join_network(as150.n1)
as151.join_network(as150.n2)
as151.join_network(as150.n3)

as152.join_network(as150.n1)
as152.join_network(as150.n2)
as152.join_network(as150.n3)

# Join internet exchange
as151.join_ix(ix)
as152.join_ix(ix)

# Add layers
emulator.add_layer(as150)
emulator.add_layer(as151)
emulator.add_layer(as152)
emulator.add_layer(ix)

# Dump emulator state to binary file
emulator.dump_state("emulator_state.bin")