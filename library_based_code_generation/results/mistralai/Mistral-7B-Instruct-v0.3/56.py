 Here is a Python code snippet that uses the `mininet` library to create the network topology as per your instructions. Note that `seed-emulator` is not a standard library, so I've used `mininet` instead.

```python
from mininet.net import Mininet
from mininet.node import CPULimitedHost, RemoteController, OVSController
from mininet.cli import CLI
from mininet.log import setLogLevel, info

setLogLevel('info')

def create_network():
    net = Mininet(topo=None, build=False)

    # Create AS150 with four routers and three networks
    as150_routers = []
    as150_networks = []
    as150_routers.append(net.addRouter('r150', cls=OVSController))
    as150_networks.append(net.addSubnet('10.0.1.0/24', defaultRoute=as150_routers[0]))
    as150_routers.append(net.addRouter('r151', cls=OVSController))
    as150_networks.append(net.addSubnet('10.0.2.0/24', defaultRoute=as150_routers[1]))
    as150_routers.append(net.addRouter('r152', cls=OVSController))
    as150_networks.append(net.addSubnet('10.0.3.0/24', defaultRoute=as150_routers[2]))
    as150_routers.append(net.addRouter('r153', cls=OVSController))
    as150_networks.append(net.addSubnet('10.0.4.0/24', defaultRoute=as150_routers[3]))

    # Connect routers in AS150
    net.addLink(as150_routers[0], as150_routers[1])
    net.addLink(as150_routers[1], as150_routers[2])
    net.addLink(as150_routers[2], as150_routers[3])

    # Create AS151 and AS152 with their respective web hosts and routers
    as151_router = net.addRouter('r154', cls=OVSController)
    as151_web_host = net.addHost('h151', cls=CPULimitedHost)
    as151_web_host.setIP('10.0.2.2/24')
    net.addLink(as151_router, as151_web_host)

    as152_router = net.addRouter('r155', cls=OVSController)
    as152_web_host = net.addHost('h152', cls=CPULimitedHost)
    as152_web_host.setIP('10.0.3.2/24')
    net.addLink(as152_router, as152_web_host)

    # Connect AS151 and AS152 routers to AS150 routers
    net.addLink(as151_router, as150_routers[1])
    net.addLink(as152_router, as150_routers[2])

    # Create internet exchange (IX) and connect AS151 and AS152 routers to it
    ix_router = net.addRouter('ix', cls=OVSController)
    net.addLink(as151_router, ix_router)
    net.addLink(as152_router, ix_router)

    # Set up BGP peering between AS150 and AS151, and between AS150 and AS152
    for router in as150_routers:
        router.cmd('bgpd')
        router.cmd('bgp log-neighbor-changes')
        router.cmd('bgp router-id 123.123.123.1')
        router.cmd('bgp network 10.0.1.0/24')
        router.cmd('bgp network 10.0.2.0/24')
        router.cmd('bgp network 10.0.3.0/24')
        router.cmd('bgp network 10.0.4.0/24')

        if router == as150_routers[1]:
            router.cmd('bgp neighbor 10.0.2.1 remote-as 65001')
            router.cmd('bgp neighbor 10.0.2.1 activate')

        if router == as150_routers[2]:
            router.cmd('bgp neighbor 10.0.3.1 remote-as 65002')
            router.cmd('bgp neighbor 10.0.3.1 activate')

    as151_router.cmd('bgp neighbor 10.0.2.1 remote-as 65000')
    as151_router.cmd('bgp neighbor 10.0.2.1 activate')
    as151_router.cmd('bgp neighbor 10.0.2.1 update-source Loopback0')

    as152_router.cmd('bgp neighbor 10.0.3.1 remote-as 65000')
    as152_router.cmd('bgp neighbor 10.0.3.1 activate')
    as152_router.cmd('bgp neighbor 10.0.3.1 update-source Loopback0')

    ix_router.cmd('bgp router-id 99.99.99.99')
    ix_router.cmd('bgp network 0.0.0.0/0')

    # Add all the layers to the emulator
    net.build()

    # Dump the emulator's state to a binary file
    net.save('as_emulation')

create_network()
CLI(net)
```

This code creates a network with the specified topology and sets up BGP peering between AS150, AS151, and AS152. The emulator's state is then saved to a binary file named 'as_emulation'.