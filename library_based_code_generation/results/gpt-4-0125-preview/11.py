```python
from seedemu import *
from seedemu.layers import Base, Routing, Ebgp
import pickle

# Create the base layer
base = Base()

# Autonomous System 1
as1 = base.createAutonomousSystem(100)
r1 = as1.createRouter('r1')
net100 = as1.createNetwork('net100')
for i in range(5):
    host = as1.createHost(f'h{i+1}')
    net100.connect(host)
net100.connect(r1)

# Autonomous System 2
as2 = base.createAutonomousSystem(200)
net200 = as2.createNetwork('net200')
net201 = as2.createNetwork('net201')
net202 = as2.createNetwork('net202')
r2 = as2.createRouter('r2')
r3 = as2.createRouter('r3')
r4 = as2.createRouter('r4')
net200.connect(r2)
net201.connect(r3)
net202.connect(r4)

# Autonomous System 3
as3 = base.createAutonomousSystem(300)
net300 = as3.createNetwork('net300')
r5 = as3.createRouter('r5')
r6 = as3.createRouter('r6')
net300.connect(r5)
net300.connect(r6)

# Create the routing layer
routing = Routing()

# Create the eBGP layer
ebgp = Ebgp()

# Add private peering
ebgp.addPrivatePeering(100, 200, 'net100', 'net200')
ebgp.addPrivatePeering(200, 300, 'net201', 'net300')
ebgp.addPrivatePeering(300, 100, 'net300', 'net100')

# Create the emulator
emu = Emulator()

# Add the layers to the emulator
emu.addLayer(base)
emu.addLayer(routing)
emu.addLayer(ebgp)

# Dump the emulator state to a binary file
with open('emulator_state.bin', 'wb') as f:
    pickle.dump(emu, f)
```