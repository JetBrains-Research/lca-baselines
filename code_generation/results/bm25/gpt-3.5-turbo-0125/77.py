emulator = makeEmulatorBaseWith10StubASAndHosts()

ethereum_service = createEthereumService(saveState=True, override=True)

pow_blockchain = createBlockchain("POW")
poa_blockchain = createBlockchain("POA")

pow_nodes = createNodes(pow_blockchain, 4)
poa_nodes = createNodes(poa_blockchain, 4)

setAsBootNode(pow_blockchain, getNodeByAsnAndName(pow_blockchain, 1, "node1"))
setAsBootNode(pow_blockchain, getNodeByAsnAndName(pow_blockchain, 2, "node2"))
setAsBootNode(poa_blockchain, getNodeByAsnAndName(poa_blockchain, 1, "node1"))
setAsBootNode(poa_blockchain, getNodeByAsnAndName(poa_blockchain, 2, "node2"))

createAccount(getNodeByAsnAndName(pow_blockchain, 3, "node3"), balance=100)
createAccount(getNodeByAsnAndName(poa_blockchain, 3, "node3"), balance=100)

setCustomGeth(getNodeByAsnAndName(pow_blockchain, 4, "node4"), customOptions="--rpc --rpcport 8545")
setCustomGeth(getNodeByAsnAndName(poa_blockchain, 4, "node4"), customOptions="--rpc --rpcport 8545")

enableOn(getNodeByAsnAndName(pow_blockchain, 1, "node1"), "HTTP")
enableOn(getNodeByAsnAndName(pow_blockchain, 1, "node1"), "WebSocket")
setCustomGeth(getNodeByAsnAndName(pow_blockchain, 1, "node1"), customGethBinary="custom_geth_binary")

customizeNodeDisplayName(getVirtualNodes(emulator), "node1", "CustomNode1")
customizeNodeDisplayName(getVirtualNodes(emulator), "node2", "CustomNode2")
customizeNodeDisplayName(getVirtualNodes(emulator), "node3", "CustomNode3")
customizeNodeDisplayName(getVirtualNodes(emulator), "node4", "CustomNode4")

bindVirtualNodesToPhysicalNodes(emulator, filters)

addLayerToEmulator(emulator, ethereum_service)

saveComponentToFile(emulator, "ethereum_component.json")

compileEmulatorWithDocker(emulator, outputDirectory="output_directory")