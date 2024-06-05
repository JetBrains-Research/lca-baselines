 Here is the Python code based on your instructions. I've used the `docker` and `web3.py` libraries for this task. Please make sure to install these libraries before running the code.

```python
from eth_docker import docker, utils
from eth_docker.utils import get_default_geth_options
from eth_docker.utils import get_default_geth_poa_options
from eth_docker.utils import get_default_geth_pow_options

# Create the emulator with 10 stub Autonomous Systems (AS) and hosts
emulator = docker.makeEmulatorBaseWith10StubASAndHosts()

# Create Ethereum service with saveState set to True and override set to True
emulator.ethereum = {'saveState': True, 'override': True}

# Create two blockchains, one based on Proof of Work (POW) and the other on Proof of Authority (POA)
blockchain_pow = emulator.createBlockchain()
blockchain_poa = emulator.createBlockchain()

# Create four nodes for each blockchain
for i in range(4):
    if i < 2:
        # Set the first two nodes of each blockchain as bootnodes and start mining on them
        blockchain_pow.get_eth_nodes()[i] = emulator.connect_to_geth_pow(get_default_geth_pow_options())
        blockchain_poa.get_eth_nodes()[i] = emulator.connect_to_geth_poa(get_default_geth_poa_options())
    elif i == 2:
        # Create accounts with a certain balance
        account_balance = web3.toWei(100, 'ether')
        blockchain_pow.get_eth_nodes()[i] = emulator.connect_to_geth(get_default_geth_options(account_balance))
        blockchain_poa.get_eth_nodes()[i] = emulator.connect_to_geth(get_default_geth_options(account_balance))
    elif i == 3:
        # Custom geth command options
        custom_options = {'custom_options': ['--rpcport', '8546', '--rpcaddr', '0.0.0.0', '--wsport', '8547', '--wsaddr', '0.0.0.0']}
        blockchain_pow.get_eth_nodes()[i] = emulator.setCustomGeth(blockchain_pow.get_eth_nodes()[i], custom_options)
        blockchain_poa.get_eth_nodes()[i] = emulator.setCustomGeth(blockchain_poa.get_eth_nodes()[i], custom_options)
    else:
        # Set custom geth binary file on one of the nodes
        custom_geth_binary = '/path/to/custom/geth'
        blockchain_pow.get_eth_nodes()[3] = emulator.setCustomGeth(blockchain_pow.get_eth_nodes()[3], {'custom_geth_binary': custom_geth_binary})
        blockchain_poa.get_eth_nodes()[3] = emulator.setCustomGeth(blockchain_poa.get_eth_nodes()[3], {'custom_geth_binary': custom_geth_binary})

    # Enable HTTP and WebSocket connections on certain nodes
    blockchain_pow.get_eth_nodes()[i].enableOn('http')
    blockchain_pow.get_eth_nodes()[i].enableOn('ws')
    blockchain_poa.get_eth_nodes()[i].enableOn('http')
    blockchain_poa.get_eth_nodes()[i].enableOn('ws')

    # Customize the display names of the nodes for visualization purposes
    blockchain_pow.get_eth_nodes()[i].displayName = f'Blockchain POW Node {i+1}'
    blockchain_poa.get_eth_nodes()[i].displayName = f'Blockchain POA Node {i+1}'

# Bind the virtual nodes to physical nodes using filters
emulator.bindVirtualNodes(blockchain_pow.getVirtualNodes(), blockchain_pow.getBootNodes())
emulator.bindVirtualNodes(blockchain_poa.getVirtualNodes(), blockchain_poa.getBootNodes())

# Add the Ethereum layer to the emulator
emulator.addLayer('ethereum', [blockchain_pow, blockchain_poa])

# Save the component to a file
emulator.save('emulator.json')

# Compile the emulator with Docker and save the output to a directory
utils.compileDockerCompose(emulator, 'emulator', 'output_directory')
```