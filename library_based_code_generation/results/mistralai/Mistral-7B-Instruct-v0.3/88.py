 Here is the Python code that fulfills your requirements using the seedemu library:

```python
import os
import seedemu as sm
import ethereum
import ethereum.eth.vm as vm

# Create an emulator base with 10 Stub AS and 3 hosts per stub AS
emulator = sm.Emulator()
emulator.create_network(10, 3)

# Ethereum service layer and sub-layer
eth_layer = emulator.add_service_layer('ethereum')
pos_blockchain = eth_layer.add_sub_layer('pos')
pos_blockchain.consensus_mechanism = vm.POW_POS
pos_blockchain.name = 'pos'
pos_blockchain.terminal_total_difficulty = 30

# Create blockchain virtual nodes, Docker container labels, and enable Geth communication
for as_num, as_ in enumerate(emulator.as_list()):
    for host_num, host in enumerate(as_.host_list()):
        virtual_node = pos_blockchain.add_virtual_node(host)
        virtual_node.docker_container_label = f'ethereum-node-{as_num}-{host_num}'
        virtual_node.geth_http_enabled = True

        # Set specific hosts as BeaconSetupNode, BootNode, and validator nodes with different conditions
        if as_num == 0 and host_num == 0:
            virtual_node.beacon_setup_node = True
            virtual_node.boot_node = True
        elif as_num == 1 and host_num == 1:
            virtual_node.validator_node = True
        elif as_num == 2 and host_num == 2:
            virtual_node.validator_node = True

        # Customize the display names of the nodes and bind the virtual node to the physical node
        virtual_node.display_name = f'Node {as_num}-{host_num}'
        virtual_node.physical_node = host

# Add the Ethereum layer to the emulator, render it, and compile it with Docker
emulator.add_layer(eth_layer)
emulator.render()

# Save the output in the './output' directory and override existing files
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
emulator.compile(output_dir, overwrite=True)

# Enable internetMap and etherView
emulator.internetmap_enabled = True
emulator.etherview_enabled = True
```

This code creates an emulator with the specified configuration, adds an Ethereum service layer and a sub-layer for the blockchain, sets the terminal total difficulty, creates virtual nodes for each host, sets specific hosts as BeaconSetupNode, BootNode, and validator nodes, customizes the display names, binds the virtual nodes to the physical nodes, adds the Ethereum layer to the emulator, renders it, compiles it with Docker, and saves the output in the './output' directory. The existing files will be overridden.