from seedemu import *

emulator = makeEmulatorBaseWith10StubASAndHosts(3)

ethereum_layer = createBlockchain("pos", consensus_mechanism="POS", terminal_total_difficulty=30)

for asn in range(1, 11):
    stub_as = get_as_by_asn(asn)
    for host in stub_as.hosts:
        blockchain_node = createNode("BlockchainNode")
        docker_label = createNode("DockerContainerLabel")
        connect_to_geth(blockchain_node, host, communication_protocol="http")
        if host.name == "BeaconSetupNode":
            setVirtualNode(blockchain_node, "BeaconSetupNode")
        elif host.name == "BootNode":
            setVirtualNode(blockchain_node, "BootNode")
        elif host.name == "ValidatorNode":
            setVirtualNode(blockchain_node, "ValidatorNode")
        customize_display_name(blockchain_node, f"{host.name}_BlockchainNode")
        bind_virtual_node_to_physical_node(blockchain_node, host)

addLayerToEmulator(emulator, ethereum_layer)
renderEmulator(emulator)
compileEmulatorWithDocker(emulator, internetMap=True, etherView=True, output_directory="./output", override_existing_files=True)