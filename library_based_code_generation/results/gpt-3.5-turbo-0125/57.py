import seedemu

# Create autonomous systems
AS150 = seedemu.AutonomousSystem("AS150")
AS2 = seedemu.AutonomousSystem("AS2")
AS151 = seedemu.AutonomousSystem("AS151")

# Create routers and networks for AS150
router150 = seedemu.Router("Router150")
network150 = seedemu.Network("Network150")
AS150.add_router(router150)
AS150.add_network(network150)

# Create routers and networks for AS2
router2_1 = seedemu.Router("Router2_1")
router2_2 = seedemu.Router("Router2_2")
network2 = seedemu.Network("Network2")
AS2.add_router(router2_1)
AS2.add_router(router2_2)
AS2.add_network(network2)

# Create routers and networks for AS151
router151 = seedemu.Router("Router151")
network151 = seedemu.Network("Network151")
AS151.add_router(router151)
AS151.add_network(network151)

# Connect AS150 and AS2 through IX 100
AS150.connect(AS2, seedemu.InternetExchange("IX100"))

# Connect AS2 and AS151 through IX 101
AS2.connect(AS151, seedemu.InternetExchange("IX101"))

# Create BGP attacker component
attacker = seedemu.BGPAttacker("Attacker", prefix="AS151_prefix")
AS2.add_attacker(attacker)

# Establish private peering relationships
AS150.add_private_peering(AS2, "IX100")
AS151.add_private_peering(AS2, "IX101")
attacker.add_private_peering(AS2, "IX100")

# Render and compile simulation with Docker
seedemu.render_simulation()
seedemu.compile_simulation()