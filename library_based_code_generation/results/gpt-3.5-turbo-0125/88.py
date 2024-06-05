from seedemu import SeedEmu

emu = SeedEmu()
emu.create_stub_as(10, num_hosts=3)

ethereum_layer = emu.create_ethereum_service_layer()
pos_blockchain = ethereum_layer.create_blockchain("pos", consensus_mechanism="POS", total_difficulty=30)

for host in emu.hosts:
    blockchain_node = host.create_blockchain_virtual_node()
    docker_label = host.create_docker_container_label()
    host.enable_geth_communication()

    if host.name == "BeaconSetupNode":
        blockchain_node.set_condition("BeaconSetupNode")
    elif host.name == "BootNode":
        blockchain_node.set_condition("BootNode")
    elif host.name.startswith("validator"):
        blockchain_node.set_condition("ValidatorNode")

    blockchain_node.set_display_name(f"My{host.name}Node")
    blockchain_node.bind_to_physical_node(host)

emu.add_layer(ethereum_layer)
emu.render()
emu.compile_with_docker(internetMap=True, etherView=True, output_dir="./output", override=True)