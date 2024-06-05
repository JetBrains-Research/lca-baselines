 ```python
import seedeemu

emulator = seedeemu.Emulator(base_name="emulator", num_as=10, seed=1234)

eth_service = seedeemu.services.ethereum.EthereumService(emulator)
eth_service.add_blockchain(name="pos", consensus="POS", terminal_total_difficulty=30)

for asn in emulator.asns:
    for host in asn.hosts:
        node = eth_service.add_virtual_node(host)
        node.docker_labels["type"] = "geth"
        node.geth_http = True

        if host.ip == "10.0.0.1":
            node.set_beacon_setup_node()
            node.display_name = "Beacon Setup Node"
        elif host.ip == "10.0.0.2":
            node.set_boot_node()
            node.display_name = "Boot Node"
        elif host.ip == "10.0.0.3":
            node.set_validator_node()
            node.display_name = "Validator Node"
        else:
            node.display_name = f"Virtual Node {host.ip}"

        node.bind_to_physical_node()

emulator.add_service(eth_service)
emulator.render(output_dir="./output", override=True, internet_map=True, etherview=True)
emulator.compile(output_dir="./output")
```