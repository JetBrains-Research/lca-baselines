  Here is the Python code that implements the given instructions:
```
from seed_emulator import *

# Create an emulation with base, routing, eBGP, iBGP, OSPF, and web service layers
emulation = makeEmulatorBaseWith5StubASAndHosts()

# Define a function to create a stub autonomous system with a web server and a router that join a network and an internet exchange
def create_stub_as(asn, name, network, internet_exchange):
    # Create a stub autonomous system
    as_ = createAutonomousSystem(asn, name)
    # Create a router for the stub autonomous system
    router = as_.createRouter()
    # Join the router to the network
    router.joinNetwork(network)
    # Join the router to the internet exchange
    router.joinInternetExchange(internet_exchange)
    # Create a web server for the stub autonomous system
    web_server = as_.createWebServer()
    # Return the stub autonomous system
    return as_

# Create three internet exchanges
exchanges = [createInternetExchange(f"exchange{i}") for i in range(3)]

# Create multiple stub autonomous systems that join these exchanges
stub_ases = [create_stub_as(f"stub_as{i}", f"Stub AS {i}", "network{i}", exchanges[i % 3]) for i in range(10)]

# Create two autonomous systems with routers that join different networks and internet exchanges
as1 = createAutonomousSystem("as1", "AS 1")
as2 = createAutonomousSystem("as2", "AS 2")

# Join the routers of the two autonomous systems to different networks and internet exchanges
as1.createRouter().joinNetwork("network1")
as1.createRouter().joinInternetExchange(exchanges[0])
as2.createRouter().joinNetwork("network2")
as2.createRouter().joinInternetExchange(exchanges[1])

# Define private peerings between different autonomous systems
for as_ in stub_ases:
    as_.createPrivatePeering(as1)
    as_.createPrivatePeering(as2)

# Add a BGP attacker component that hijacks certain prefixes and joins an internet exchange
attacker = BgpAttackerComponent("attacker", "BGP Attacker")
attacker.setInternetExchange(exchanges[2])
attacker.setHijackPrefixes(["10.0.0.0/8", "172.16.0.0/12"])

# Merge the BGP attacker with the emulator
emulation.merge(attacker)

# Compile the new emulator using Docker and output the result to a specified directory
emulation.compile(output_dir="./emulation")
```
Note that this code assumes that you have already installed the `seed_emulator` library and that you have the necessary dependencies installed. Additionally, this code assumes that you have a Docker environment set up and that you have the necessary permissions to run Docker commands.