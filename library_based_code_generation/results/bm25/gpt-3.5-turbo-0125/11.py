import seed_emulator

emulation = seed_emulator.createEmulation()

base_layer = seed_emulator.makeEmulatorBaseWith5StubASAndHosts()
routing_layer = seed_emulator.createAutonomousSystem()
eBGP_layer = seed_emulator.createAutonomousSystem()

base_as1 = seed_emulator.AutonomousSystem()
base_as2 = seed_emulator.AutonomousSystem()
base_as3 = seed_emulator.AutonomousSystem()

base_as1.__configureAutonomousSystem(seed_emulator.makeEmulatorBaseWith5StubASAndHosts(5))
base_as2.__configureAutonomousSystem(seed_emulator.makeEmulatorBaseWith10StubASAndHosts(3))
base_as3.__configureAutonomousSystem(seed_emulator.makeEmulatorBaseWith10StubASAndHosts(2))

network1 = seed_emulator.createNetwork()
network2 = seed_emulator.createNetwork()
network3 = seed_emulator.createNetwork()

base_as1.__joinNetwork(network1)
base_as2.__joinNetwork(network2)
base_as3.__joinNetwork(network3)

routing_layer.shouldMerge(base_as1)
routing_layer.shouldMerge(base_as2)
routing_layer.shouldMerge(base_as3)

eBGP_layer.shouldMerge(routing_layer)

emulation.setBaseSystem(base_layer)
emulation.setAutonomousSystem(routing_layer)
emulation.setAutonomousSystem(eBGP_layer)

seed_emulator.dumpEmulatorStateToFile(emulation, "emulator_state.bin")