  from seed_emulator import *

# Create the emulation
emulation = createEmulation()

# Create the base layer
base = makeEmulatorBaseWith5StubASAndHosts(emulation)

# Create the routing layer
routing = createNetwork(base, "routing")

# Create the eBGP layer
ebgp = createNetwork(base, "ebgp")

# Create the first autonomous system
as1 = createAutonomousSystem(base, "as1")

# Create the second autonomous system
as2 = createAutonomousSystem(base, "as2")

# Create the third autonomous system
as3 = createAutonomousSystem(base, "as3")

# Add private peering between AS1 and AS2
as1.addPrivatePeering(as2)

# Add private peering between AS2 and AS3
as2.addPrivatePeering(as3)

# Add the networks to the emulation
emulation.addNetwork(routing)
emulation.addNetwork(ebgp)

# Dump the emulator state to a binary file
emulation.dump("emulation.bin")

# Print the emulation state
print(emulation)