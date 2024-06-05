 ```python
from seed_emulator import createEmulation, makeEmulatorBaseWith5StubASAndHosts, makeEmulatorBaseWith10StubASAndHosts
from seed_emulator.scion_emulator import ScionAutonomousSystem, BaseSystem, PoAServer

# Create emulation
emul = createEmulation()

# Configure base layer
emul.makeEmulatorBaseWith5StubASAndHosts()
emul.makeEmulatorBaseWith10StubASAndHosts()

as1 = emul.getAutonomousSystem('as1')
as2 = emul.getAutonomousSystem('as2')
as3 = emul.getAutonomousSystem('as3')

network1 = as1.createNetwork('10.0.0.0/24')
network2 = as2.createNetwork('20.0.0.0/24')
network3 = as2.createNetwork('30.0.0.0/24')
network4 = as3.createNetwork('40.0.0.0/24')

as1_router1 = as1.getRouters()[0]
as1_router1.joinNetwork(network1)

as2_router1 = as2.getRouters()[0]
as2_router1.joinNetwork(network2)

as2_router2 = as2.getRouters()[1]
as2_router2.joinNetwork(network3)

as3_router1 = as3.getRouters()[0]
as3_router1.joinNetwork(network4)

as3_router2 = as3.getRouters()[1]
as3_router2.joinNetwork(network4)

# Configure routing layer
as1.setBaseSystem(BaseSystem(as1_router1))
as2.setBaseSystem(BaseSystem(as2_router1))
as3.setBaseSystem(BaseSystem(as3_router1))

# Configure eBGP layer
as1.createAutonomousSystem('as4', PoAServer)
as4 = emul.getAutonomousSystem('as4')
as4.setBaseSystem(BaseSystem(as1_router1))

as4.joinNetwork(network1)

as4_router = as4.getRouters()[0]
as4_router.joinNetwork(network2)

as2.createAutonomousSystem('as5', PoAServer)
as5 = emul.getAutonomousSystem('as5')
as5.setBaseSystem(BaseSystem(as2_router1))

as5.joinNetwork(network2)

as5_router = as5.getRouters()[0]
as5_router.joinNetwork(network4)

# Add layers to the emulator
emul.addLayer(as1)
emul.addLayer(as2)
emul.addLayer(as3)
emul.addLayer(as4)
emul.addLayer(as5)

# Dump emulator state to a binary file
emul.dump('emulator_state.bin')
```