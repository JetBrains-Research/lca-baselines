  from seed_emulator import Emulator

# Create the base layer
base = Emulator.Layer("base")

# Create the first autonomous system
as1 = base.AutonomousSystem(1, "as1")

# Create the router and network for the first autonomous system
router1 = as1.Router("router1")
network1 = as1.Network("network1")

# Create the hosts for the first autonomous system
host1 = network1.Host("host1")
host2 = network1.Host("host2")
host3 = network1.Host("host3")
host4 = network1.Host("host4")
host5 = network1.Host("host5")

# Create the second autonomous system
as2 = base.AutonomousSystem(2, "as2")

# Create the routers and networks for the second autonomous system
router2 = as2.Router("router2")
network2 = as2.Network("network2")
router3 = as2.Router("router3")
network3 = as2.Network("network3")

# Create the third autonomous system
as3 = base.AutonomousSystem(3, "as3")

# Create the routers and networks for the third autonomous system
router4 = as3.Router("router4")
network4 = as3.Network("network4")
router5 = as3.Router("router5")
network5 = as3.Network("network5")

# Add private peering between the autonomous systems
as1.add_private_peering(as2)
as1.add_private_peering(as3)

# Add the layers to the emulator
emulator = Emulator()
emulator.add_layer(base)

# Dump the emulator state to a binary file
emulator.dump("emulator.bin")