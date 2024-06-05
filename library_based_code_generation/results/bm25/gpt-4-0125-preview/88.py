```python
from seedemu import *
from seedemu.layers import Base, Ethereum

# Create the emulator base with 10 Stub ASes, each with 3 hosts
base = makeEmulatorBaseWith10StubASAndHosts(3)

# Create the Ethereum service layer
ethereum = Ethereum()

# Create a blockchain sub-layer with the name "pos" and consensus mechanism set to POS
blockchain = ethereum.createBlockchain('pos', 'POS')

# Set the terminal total difficulty of the blockchain
blockchain.setTerminalTotalDifficulty(30)

# Iterate through each AS and its hosts to configure the Ethereum nodes
for asn in range(100, 110):
    for host_number in range(1, 4):
        host_name = f"host{host_number}"
        host = base.getNodeByAsnAndName(asn, host_name)

        # Create a blockchain virtual node for each host
        vnode = blockchain.createNode()

        # Set Docker container label
        vnode.setLabel(f"{asn}-{host_name}-eth")

        # Enable Geth to communicate with the geth node via http
        vnode.enableHttp()

        # Bind the virtual node to the physical node
        blockchain.bind(vnode, host)

        # Customize display names and set specific roles
        if host_number == 1:
            vnode.setDisplayName(f"BeaconSetupNode-{asn}")
            blockchain.markAsBeaconSetupNode(vnode)
        elif host_number == 2:
            vnode.setDisplayName(f"BootNode-{asn}")
            blockchain.markAsBootNode(vnode)
        elif host_number == 3:
            vnode.setDisplayName(f"ValidatorNode-{asn}")
            blockchain.markAsValidator(vnode)

# Add the Ethereum layer to the emulator
base.addLayer(ethereum)

# Render the emulator and compile it with Docker
emulator = Emulator()
emulator.render(base)
emulator.compile(outputDirectory='./output', overwrite=True, enableInternetMap=True, enableEtherView=True)
```