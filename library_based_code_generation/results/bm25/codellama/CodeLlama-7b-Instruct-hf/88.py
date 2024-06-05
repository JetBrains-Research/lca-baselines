  Here is the Python code based on the given instructions:
```
from seedemu import *

# Create an emulator base with 10 Stub AS and 3 hosts per stub AS
emulator_base = makeEmulatorBaseWith10StubASAndHosts()

# Create an Ethereum service layer and a sub-layer of it, a blockchain with the name "pos" and consensus mechanism set to POS
ethereum_service_layer = makeEthereumServiceLayer()
blockchain = createBlockchain(name="pos", consensus_mechanism="POS")
ethereum_service_layer.add_blockchain(blockchain)

# Set the terminal total difficulty of the blockchain to 30
blockchain.set_terminal_total_difficulty(30)

# For each host in the AS, create a blockchain virtual node, a Docker container label, and enable Geth to communicate with the geth node via http
for host in emulator_base.hosts:
    virtual_node = setVirtualNode(host, blockchain)
    docker_container_label = "ethereum-node-" + host.name
    connect_to_geth(virtual_node, docker_container_label)

# Set specific hosts as BeaconSetupNode, BootNode, and validator nodes with different conditions
beacon_setup_node = getNodeByAsnAndName(emulator_base, 1, "beacon-setup-node")
boot_node = getNodeByAsnAndName(emulator_base, 2, "boot-node")
validator_nodes = [getNodeByAsnAndName(emulator_base, 3, "validator-node-1"),
                   getNodeByAsnAndName(emulator_base, 4, "validator-node-2"),
                   getNodeByAsnAndName(emulator_base, 5, "validator-node-3")]

# Customize the display names of the nodes and bind the virtual node to the physical node
for node in [beacon_setup_node, boot_node] + validator_nodes:
    node.display_name = _nodeRoleToString(node.role)
    virtual_node = getVirtualNode(node)
    physical_node = get_all_accounts_with_node_info(node)
    virtual_node.bind_to_physical_node(physical_node)

# Add the Ethereum layer to the emulator, render it, and compile it with Docker with internetMap and etherView enabled
ethereum_layer = makeEthereumLayer(emulator_base, ethereum_service_layer)
render_emulator(emulator_base)
compile_emulator(emulator_base, docker_image_name="ethereum-emulator", internet_map=True, ether_view=True)

# Save the output to the './output' directory and override existing files
save_emulator(emulator_base, "./output")
```
Note that this code uses the `seedemu` library to create an emulator base with 10 Stub AS and 3 hosts per stub AS, create an Ethereum service layer and a sub-layer of it, a blockchain with the name "pos" and consensus mechanism set to POS, set the terminal total difficulty of the blockchain to 30, create blockchain virtual nodes, enable Geth to communicate with the geth node via http, set specific hosts as BeaconSetupNode, BootNode, and validator nodes with different conditions, customize the display names of the nodes and bind the virtual node to the physical node, add the Ethereum layer to the emulator, render it, and compile it with Docker with internetMap and etherView enabled, and save the output to the './output' directory and override existing files.