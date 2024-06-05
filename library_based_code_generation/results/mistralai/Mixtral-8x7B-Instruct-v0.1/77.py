 ```python
from emulator_suite.emulator import Emulator
from emulator_suite.services import EthereumService
from emulator_suite.filters import Filter
from emulator_suite.components import Component

# Create emulator with 10 stub AS and hosts
emulator = Emulator(as_count=10, host_count=10)

# Create Ethereum service with saveState and override set to True
ethereum_service = EthereumService(save_state=True, override=True)
emulator.add_service(ethereum_service)

# Create POW blockchain with 4 nodes, set first two as bootnodes and start mining
pow_blockchain = ethereum_service.add_blockchain(name="POW", consensus="pow")
for i in range(4):
    node = pow_blockchain.add_node()
    if i < 2:
        node.set_bootnode()
    if i == 0:
        node.start_mining()

# Create POA blockchain with 4 nodes, set first two as bootnodes and start mining
poa_blockchain = ethereum_service.add_blockchain(name="POA", consensus="poa")
for i in range(4):
    node = poa_blockchain.add_node()
    if i < 2:
        node.set_bootnode()
    if i == 1:
        node.start_mining()

# Create accounts with balance on third node of each blockchain
for blockchain in [pow_blockchain, poa_blockchain]:
    third_node = blockchain.nodes[2]
    third_node.create_account(balance=1000000000000000000)

# Set custom geth command options for fourth node of each blockchain
for blockchain in [pow_blockchain, poa_blockchain]:
    fourth_node = blockchain.nodes[3]
    fourth_node.set_geth_options(
        extra_opts=["--syncmode", "full", "--gcmode", "archive"]
    )

# Enable HTTP and WebSocket connections on certain nodes
for blockchain in [pow_blockchain, poa_blockchain]:
    first_node = blockchain.nodes[0]
    first_node.enable_http()
    first_node.enable_ws()

# Set custom geth binary file on one of the nodes
first_node_in_pow_blockchain = pow_blockchain.nodes[0]
first_node_in_pow_blockchain.set_geth_binary("path/to/custom/geth")

# Customize the display names of the nodes for visualization purposes
for blockchain in [pow_blockchain, poa_blockchain]:
    for node in blockchain.nodes:
        node.set_display_name(f"{blockchain.name}_{node.index}")

# Bind the virtual nodes to physical nodes using filters
filter_pow_blockchain = Filter(chain_name="POW")
filter_poa_blockchain = Filter(chain_name="POA")
emulator.bind_filters([filter_pow_blockchain, filter_poa_blockchain])

# Add the Ethereum layer to the emulator and save the component to a file
ethereum_component = Component(emulator)
ethereum_component.save("path/to/ethereum/component")

# Compile the emulator with Docker and save the output to a directory
emulator.compile(output_dir="path/to/output/directory")
```