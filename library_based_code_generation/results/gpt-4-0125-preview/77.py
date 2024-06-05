```python
from emuvim.api.osm.emulator import SimpleEPC
from emuvim.dcemulator.net import Containernet
from emuvim.dcemulator.node import DockerStarterAutoNetkit
from mininet.log import setLogLevel
from mininet.node import RemoteController
import json

def create_emulator():
    setLogLevel('info')

    net = Containernet(controller=RemoteController, link=TCLink)
    net.addController('c0')

    # Create 10 Autonomous Systems (AS) and hosts
    as_list = []
    for i in range(1, 11):
        as_name = "as%d" % i
        as_list.append(net.addDockerStarterAutoNetkit(as_name, dimage="ubuntu:trusty"))

    # Ethereum service configuration
    ethereum_service = {
        "saveState": True,
        "override": True
    }

    # Create two blockchains: POW and POA
    blockchains = {
        "POW": [],
        "POA": []
    }

    for bc_type in blockchains.keys():
        for i in range(1, 5):
            node_name = "%s_node%d" % (bc_type, i)
            display_name = "%s Node %d" % (bc_type, i)
            node = net.addDockerStarterAutoNetkit(node_name, dimage="ethereum/client-go:latest")
            node.setDisplayName(display_name)
            if i <= 2:  # Bootnodes
                node.addService("bootnode")
                node.startMining()
            elif i == 3:  # Create accounts with balance
                node.addService("account", balance="1000")
            else:  # Custom geth command options
                node.addService("geth", options="--verbosity 4")
            if i == 1 or i == 4:  # Enable HTTP and WebSocket for the first and fourth nodes
                node.enableHTTP()
                node.enableWebSocket()
            if i == 2:  # Set custom geth binary file for the second node
                node.setGethBinary("/usr/local/bin/geth-custom")
            blockchains[bc_type].append(node)

    # Bind virtual nodes to physical nodes using filters
    for bc_type, nodes in blockchains.items():
        for node in nodes:
            node.setCPUs(1)
            node.setMemory("512M")
            node.bindToPhysicalNode("physicalNodeName")

    # Add Ethereum layer to the emulator
    net.addLink(as_list[0], blockchains["POW"][0])
    net.addLink(as_list[1], blockchains["POA"][0])

    # Save the component to a file
    with open("emulator_component.json", "w") as f:
        json.dump(ethereum_service, f)

    # Compile the emulator with Docker
    net.compileDockerImage("emulator_image", "./Dockerfile")

    # Save the output to a directory
    net.saveOutput("/path/to/output_directory")

    net.start()
    net.CLI()
    net.stop()

if __name__ == '__main__':
    create_emulator()
```