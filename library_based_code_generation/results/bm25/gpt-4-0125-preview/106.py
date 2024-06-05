```python
from seedemu.core import Emulator, Binding, Filter
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, WebServiceLayer
from seedemu.services import WebServer, BgpAttacker
from seedemu.utilities import ASNumberGenerator

def create_stub_as(emulator, asn, ix):
    base = emulator.getLayer('Base')
    routing = emulator.getLayer('Routing')

    # Create autonomous system
    asys = base.createAutonomousSystem(asn)
    asys.createRouter('router')

    # Create and add a web server
    web_service = emulator.getLayer('WebService')
    web_service.install('web-server', asn)

    # Join the Internet Exchange
    routing.addInternetExchange(ix)
    routing.getInternetExchange(ix).addMember(asn)

    return asn

def main():
    emulator = Emulator()

    # Add layers
    base_layer = Base()
    routing_layer = Routing()
    ebgp_layer = Ebgp()
    ibgp_layer = Ibgp()
    ospf_layer = Ospf()
    web_service_layer = WebServiceLayer()

    emulator.addLayer(base_layer)
    emulator.addLayer(routing_layer)
    emulator.addLayer(ebgp_layer)
    emulator.addLayer(ibgp_layer)
    emulator.addLayer(ospf_layer)
    emulator.addLayer(web_service_layer)

    # Create Internet Exchanges
    ix100 = routing_layer.createInternetExchange(100)
    ix200 = routing_layer.createInternetExchange(200)
    ix300 = routing_layer.createInternetExchange(300)

    asn_gen = ASNumberGenerator()

    # Create stub ASes and join them to Internet Exchanges
    for ix in [ix100, ix200, ix300]:
        for _ in range(5):
            create_stub_as(emulator, asn_gen.next(), ix.getId())

    # Create two ASes with routers joining different networks and IXs
    as1 = create_stub_as(emulator, asn_gen.next(), ix100.getId())
    as2 = create_stub_as(emulator, asn_gen.next(), ix200.getId())

    # Define private peerings
    ebgp_layer.addPrivatePeering(as1, as2, 'router', 'router')

    # Add a BGP attacker
    attacker_asn = asn_gen.next()
    bgp_attacker = BgpAttacker(attacker_asn, '1.2.3.0/24')
    base_layer.createAutonomousSystem(attacker_asn).createRouter('attacker-router')
    routing_layer.addInternetExchange(ix300)
    routing_layer.getInternetExchange(ix300).addMember(attacker_asn)
    web_service_layer.install('bgp-attacker', attacker_asn, bgp_attacker)

    # Compile and output to a specified directory
    emulator.render()
    emulator.compile(Docker(), output_directory='/path/to/output')

if __name__ == '__main__':
    main()
```