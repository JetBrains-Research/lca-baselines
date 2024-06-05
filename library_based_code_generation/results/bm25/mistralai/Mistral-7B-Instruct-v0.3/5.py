 Here is the Python code using the seedemu library as per your instructions:

```python
from seedemu import Emulator, InternetExchange, AutonomousSystem, Router, Host, PoAServer, ScionAutonomousSystem

# Initialize the emulator
emulator = Emulator()

# Create Internet Exchange
ix_id = 1
ix = create_ix(ix_id, 'IX-1', 'Internet Exchange 1')
emulator.add_layer(ix)

# Create Autonomous Systems
as_ids = [2, 3, 4]
as_names = ['AS-2', 'AS-3', 'AS-4']
as_descriptions = ['AS-2 Description', 'AS-3 Description', 'AS-4 Description']
as_list = []
for as_id, as_name, as_description in zip(as_ids, as_names, as_descriptions):
    as_obj = create_as(as_id, as_name, as_description)
    as_list.append(as_obj)
    emulator.add_layer(as_obj)

# Create networks, routers, and hosts for each Autonomous System
for as_obj in as_list:
    network_id = as_obj.id * 100
    network_name = f'Network-{as_obj.id}'
    network_description = f'Network for AS-{as_obj.id}'
    network = create_network(network_id, network_name, network_description)
    as_obj.add_network(network)

    router_id = network_id + 1
    router_name = f'Router-{as_obj.id}'
    router_description = f'Router for AS-{as_obj.id}'
    router = create_router(router_id, router_name, router_description)
    router.add_interface(network)

    host_id = network_id + 2
    host_name = f'Host-{as_obj.id}'
    host_description = f'Host for AS-{as_obj.id}'
    host = create_host(host_id, host_name, host_description)
    host.add_interface(network)

    # Install a web service on a virtual node and bind this node to a host
    virtual_node_id = host_id + 1
    virtual_node_name = f'VirtualNode-{as_obj.id}'
    virtual_node_description = f'Virtual Node for AS-{as_obj.id}'
    virtual_node = create_virtual_node(virtual_node_id, virtual_node_name, virtual_node_description)
    host.add_virtual_node(virtual_node)

# Peer Autonomous Systems with the Internet Exchange
for as_obj in as_list:
    ix_obj = get_ix()
    as_obj.join_internet_exchange(ix_obj)

# Add all the layers to the emulator and render the emulator
emulator.render()

# Compile the internet map with Docker
emulator.compile_with_docker()

def create_ix(ix_id, name, description):
    ix = InternetExchange(ix_id, name, description)
    return ix

def create_as(as_id, name, description):
    as_obj = ScionAutonomousSystem(as_id, name, description)
    return as_obj

def create_network(network_id, name, description):
    network = AutonomousSystem.create_network(network_id, name, description)
    return network

def create_router(router_id, name, description):
    router = AutonomousSystem.create_router(router_id, name, description)
    return router

def create_host(host_id, name, description):
    host = AutonomousSystem.create_host(host_id, name, description)
    return host

def create_virtual_node(virtual_node_id, name, description):
    virtual_node = AutonomousSystem.create_virtual_node(virtual_node_id, name, description)
    return virtual_node

def get_ix():
    ix_ids = get_internet_exchange_ids()
    return get_internet_exchange(ix_ids[0])
```

Please note that this code assumes that you have already imported the necessary seedemu modules and have implemented the helper functions `get_internet_exchange_ids` and `get_internet_exchange`. These functions should return the IDs and the actual InternetExchange objects of all InternetExchange layers in the emulator, respectively.