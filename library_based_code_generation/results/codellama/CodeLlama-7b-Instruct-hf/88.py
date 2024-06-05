  from seedemu import Emulator, StubAS, Host, EthereumServiceLayer, Blockchain, ConsensusMechanism, Geth, Docker, InternetMap, EtherView

# Create an emulator base with 10 Stub AS and 3 hosts per stub AS
emulator = Emulator(num_stub_as=10, num_hosts_per_stub_as=3)

# Create an Ethereum service layer and a sub-layer of it, a blockchain with the name "pos" and consensus mechanism set to POS
ethereum_service_layer = EthereumServiceLayer(name="ethereum")
pos_blockchain = Blockchain(name="pos", consensus_mechanism=ConsensusMechanism.POS)
ethereum_service_layer.add_blockchain(pos_blockchain)

# Set the terminal total difficulty of the blockchain to 30
pos_blockchain.terminal_total_difficulty = 30

# Create a Docker container label for each host in the AS
for stub_as in emulator.stub_as_list:
    for host in stub_as.host_list:
        host.docker_container_label = f"{host.name}-container"

# Enable Geth to communicate with the geth node via http
for stub_as in emulator.stub_as_list:
    for host in stub_as.host_list:
        host.geth.enable_http_communication()

# Customize the display names of the nodes and bind the virtual node to the physical node
for stub_as in emulator.stub_as_list:
    for host in stub_as.host_list:
        host.display_name = f"{host.name}-display"
        host.bind_virtual_node_to_physical_node()

# Add the Ethereum layer to the emulator, render it, and compile it with Docker with internetMap and etherView enabled
emulator.add_layer(ethereum_service_layer)
emulator.render()
emulator.compile(Docker, internet_map=InternetMap.ENABLED, ether_view=EtherView.ENABLED)

# Save the output to the './output' directory and override existing files
emulator.save_output("./output", override=True)