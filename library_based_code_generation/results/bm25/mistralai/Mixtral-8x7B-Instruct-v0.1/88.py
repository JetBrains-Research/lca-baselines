 ```
from seedemu.lib import makeEmulatorBaseWith10StubASAndHosts, createBlockchain, setVirtualNode, getNodeByAsnAndName, connectToBlockchain, connect_to_geth_pos, get_all_accounts_with_node_info, _nodeRoleToString, markAsStub, makeStubAs, toGraphviz, to_json, resolveTo, getVirtualNode, createNode, connect_to_geth

emulator = makeEmulatorBaseWith10StubASAndHosts(seed=1234)

for asn, hosts in emulator.as_hosts.items():
    for i, host in enumerate(hosts):
        node = createNode(host)
        connect_to_geth_pos(node)
        setVirtualNode(node, f"AS{asn}_Host{i}")
        if i == 0:
            markAsStub(node)
            node.role = "BeaconSetupNode"
        elif i == 1:
            markAsStub(node)
            node.role = "BootNode"
        else:
            node.role = "Validator"
            get_all_accounts_with_node_info(node)

blockchain = createBlockchain(emulator, "pos", consensus="POS", terminalTotalDifficulty=30)

for asn, hosts in emulator.as_hosts.items():
    for i, host in enumerate(hosts):
        node = getNodeByAsnAndName(emulator, asn, f"AS{asn}_Host{i}")
        if node.role == "Validator":
            connectToBlockchain(node, blockchain)
            get_balance_with_name(node, blockchain)

toGraphviz(emulator, "./output/emulator.dot").render(directory="./output", format="png", overwrite=True)
to_json(emulator, "./output/emulator.json", overwrite=True)

emulator.addLayer(blockchain)
emulator.compile(internetMap=True, etherView=True, outputDir="./output", overwrite=True)
```