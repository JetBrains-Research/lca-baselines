  Here is the Python code based on the given instructions:
```
from emulation import makeEmulatorBaseWith5StubASAndHosts, makeEmulatorBaseWith10StubASAndHosts, connectToBlockchain, get_balance_of_account, enableOn, createHostsOnNetwork, createBlockchain, getNodeByAsnAndName, registerNodes, getVirtualNodes, connect_to_geth, getBootNodes, getMinerNodes, get_eth_nodes, __getEdgeNodes, load_eth_nodes, connect_to_geth_poa, connect_to_geth_pow, setCustomGeth, makeStubAsWithHosts

# Create an emulator with 10 stub Autonomous Systems (AS) and hosts
emulator = makeEmulatorBaseWith10StubASAndHosts()

# Create an Ethereum service with saveState set to True and override set to True
ethereum_service = connectToBlockchain(emulator, "ethereum", saveState=True, override=True)

# Create two blockchains, one based on Proof of Work (POW) and the other on Proof of Authority (POA)
blockchain_pow = createBlockchain(emulator, "pow", "ethereum")
blockchain_poa = createBlockchain(emulator, "poa", "ethereum")

# Create four nodes for each blockchain
node_pow_1 = createHostsOnNetwork(emulator, "pow", "ethereum", 4)
node_pow_2 = createHostsOnNetwork(emulator, "pow", "ethereum", 4)
node_poa_1 = createHostsOnNetwork(emulator, "poa", "ethereum", 4)
node_poa_2 = createHostsOnNetwork(emulator, "poa", "ethereum", 4)

# Set the first two nodes of each blockchain as bootnodes and start mining on them
bootnodes_pow = getNodeByAsnAndName(emulator, "pow", "ethereum", 0)
bootnodes_poa = getNodeByAsnAndName(emulator, "poa", "ethereum", 0)
enableOn(bootnodes_pow, "mining")
enableOn(bootnodes_poa, "mining")

# Create accounts with a certain balance for the third node of each blockchain
accounts_pow = getNodeByAsnAndName(emulator, "pow", "ethereum", 2)
accounts_poa = getNodeByAsnAndName(emulator, "poa", "ethereum", 2)
get_balance_of_account(accounts_pow, "ethereum", 1000)
get_balance_of_account(accounts_poa, "ethereum", 1000)

# Set custom geth command options for the fourth node of each blockchain
custom_geth_pow = setCustomGeth(emulator, "pow", "ethereum", 3)
custom_geth_poa = setCustomGeth(emulator, "poa", "ethereum", 3)

# Enable HTTP and WebSocket connections on certain nodes
enableOn(custom_geth_pow, "http")
enableOn(custom_geth_poa, "http")
enableOn(custom_geth_pow, "websocket")
enableOn(custom_geth_poa, "websocket")

# Set custom geth binary file on one of the nodes
setCustomGeth(emulator, "pow", "ethereum", 3, "geth-custom")

# Customize the display names of the nodes for visualization purposes
node_pow_1.display_name = "Node 1 (POW)"
node_pow_2.display_name = "Node 2 (POW)"
node_poa_1.display_name = "Node 1 (POA)"
node_poa_2.display_name = "Node 2 (POA)"

# Bind the virtual nodes to physical nodes using filters
filters = [
    {"type": "physical", "asn": "pow", "name": "Node 1 (POW)"},
    {"type": "physical", "asn": "pow", "name": "Node 2 (POW)"},
    {"type": "physical", "asn": "poa", "name": "Node 1 (POA)"},
    {"type": "physical", "asn": "poa", "name": "Node 2 (POA)"}
]
registerNodes(emulator, filters)

# Add the Ethereum layer to the emulator and save the component to a file
ethereum_layer = makeStubAsWithHosts(emulator, "ethereum", "ethereum")
save_component(emulator, "ethereum_layer.json")

# Compile the emulator with Docker and save the output to a directory
compile_emulator(emulator, "docker", "output_directory")
```