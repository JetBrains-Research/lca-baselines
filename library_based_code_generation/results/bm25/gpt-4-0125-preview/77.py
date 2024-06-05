```python
from emulib import makeEmulatorBaseWith10StubASAndHosts, createBlockchain, registerNodes, setCustomGeth

# Create emulator with 10 stub AS and hosts
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Create Ethereum service with saveState and override enabled
ethereumService = {
    "saveState": True,
    "override": True
}

# Create POW blockchain
powBlockchain = createBlockchain("POW", ethereumService)
powNodes = createHostsOnNetwork(powBlockchain, 4)

# Create POA blockchain
poaBlockchain = createBlockchain("POA", ethereumService)
poaNodes = createHostsOnNetwork(poaBlockchain, 4)

# Configure POW nodes
for i, node in enumerate(powNodes):
    if i < 2:
        node["role"] = "bootnode"
        node["startMining"] = True
        node["displayName"] = f"POW Bootnode {i+1}"
    elif i == 2:
        node["accounts"] = [{"balance": "100000"}]
        node["displayName"] = "POW Account Node"
    else:
        node["customGeth"] = "custom options"
        node["displayName"] = "POW Custom Node"

# Configure POA nodes
for i, node in enumerate(poaNodes):
    if i < 2:
        node["role"] = "bootnode"
        node["startMining"] = True
        node["displayName"] = f"POA Bootnode {i+1}"
    elif i == 2:
        node["accounts"] = [{"balance": "100000"}]
        node["displayName"] = "POA Account Node"
    else:
        node["customGeth"] = "custom options"
        node["displayName"] = "POA Custom Node"

# Enable HTTP and WebSocket on selected nodes
enableOn(powNodes[0], ["http", "ws"])
enableOn(poaNodes[1], ["http", "ws"])

# Set custom geth binary file on a node
setCustomGeth(powNodes[3], "/path/to/custom/geth")

# Bind virtual nodes to physical nodes using filters
# This step is highly dependent on the specific network setup and is not explicitly shown here

# Add Ethereum layer to the emulator
registerNodes(emulator, powNodes + poaNodes)

# Save the component to a file
emulator.saveToFile("emulator_component.json")

# Compile the emulator with Docker and save the output to a directory
emulator.compileWithDocker("output_directory")
```