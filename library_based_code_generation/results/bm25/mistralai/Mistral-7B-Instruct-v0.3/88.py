 Here is the Python code that follows your instructions, using the seedemu library:

```python
import seedemu as sm
import os

# Create emulator base with 10 Stub AS and 3 hosts per stub AS
emulator_base = sm.makeEmulatorBaseWith10StubASAndHosts(num_hosts_per_stub_as=3)

# Create Ethereum service layer and sub-layer
eth_service_layer = sm.createServiceLayer('ethereum')
pos_sub_layer = sm.createSubLayer(eth_service_layer, 'pos')
blockchain = sm.createBlockchain(pos_sub_layer, 'pos', consensus_mechanism='POS')
blockchain.terminal_total_difficulty = 30

# Create blockchain virtual nodes, Docker container labels, and enable Geth communication
for asn, stub_as in enumerate(emulator_base.stub_as_list):
    for host_index, host in enumerate(stub_as.host_list):
        virtual_node = sm.createNode(pos_sub_layer)
        docker_label = f'ethereum-node-{asn}-{host_index}'
        host.docker_container.labels[docker_label] = 'true'
        sm.connect_to_geth_poa(virtual_node, host.ip, host.port)

        # Set specific hosts as BeaconSetupNode, BootNode, and validator nodes with different conditions
        if asn == 0 and host_index == 0:
            virtual_node.node_roles = ['BeaconSetupNode']
        elif asn == 0 and host_index == 1:
            virtual_node.node_roles = ['BootNode']
        elif asn == 1 and host_index == 0:
            virtual_node.node_roles = ['ValidatorNode']

        # Customize the display names of the nodes and bind the virtual node to the physical node
        virtual_node.display_name = f'Node {asn}-{host_index}'
        virtual_node.physical_node = host

# Add the Ethereum layer to the emulator, render it, and compile it with Docker
emulator_base.addServiceLayer(eth_service_layer)
emulator_base.render()
emulator_base.compile(internet_map=True, ether_view=True, output_dir='./output', overwrite=True)
```

Please ensure that you have the seedemu library installed before running this code. If not, you can install it using pip:

```
pip install seedemu
```