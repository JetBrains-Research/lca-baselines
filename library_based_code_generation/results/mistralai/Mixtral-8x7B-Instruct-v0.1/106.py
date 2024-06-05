 ```python
import seed_emulator
import random

def create_stub_as(asn, web_port, router_ip, ix_peering_ips):
    asn_config = seed_emulator.ASConfig(asn)
    asn_config.add_layer('base', seed_emulator.BaseLayer)
    asn_config.add_layer('routing', seed_emulator.RoutingLayer)
    asn_config.add_layer('ebgp', seed_emulator.BGPLayer)
    asn_config.add_layer('ibgp', seed_emulator.BGPLayer, autonet=asn)
    asn_config.add_layer('ospf', seed_emulator.OSPFLayer)
    asn_config.add_layer('web', seed_emulator.WebServerLayer, port=web_port)

    router = seed_emulator.Router(router_ip)
    asn_config.add_node('r1', router)

    for ix_peering_ip in ix_peering_ips:
        router.add_link(seed_emulator.Link(ix_peering_ip, router_ip))

    return asn_config

def create_attacker(asn, ix_peering_ip):
    asn_config = seed_emulator.ASConfig(asn)
    asn_config.add_layer('base', seed_emulator.BaseLayer)
    asn_config.add_layer('routing', seed_emulator.RoutingLayer)
    asn_config.add_layer('ebgp', seed_emulator.BGPLayer)

    attacker = seed_emulator.Router(f'10.0.0.{asn}')
    asn_config.add_node('r1', attacker)

    attacker.add_link(seed_emulator.Link(ix_peering_ip, attacker.ip))

    return asn_config

ix1_peering_ips = [f'10.0.0.{i+1}' for i in range(1, 6)]
ix2_peering_ips = [f'20.0.0.{i+1}' for i in range(1, 6)]
ix3_peering_ips = [f'30.0.0.{i+1}' for i in range(1, 6)]

emulator = seed_emulator.Emulator('emulator')

as1 = create_stub_as(1, 8000, '10.0.0.1', ix1_peering_ips)
emulator.add_as(as1)

as2 = create_stub_as(2, 8001, '20.0.0.1', ix2_peering_ips)
emulator.add_as(as2)

as3 = create_stub_as(3, 8002, '30.0.0.1', ix3_peering_ips)
emulator.add_as(as3)

as4 = create_stub_as(4, 8003, '10.0.0.4', ix1_peering_ips)
emulator.add_as(as4)

as5 = create_stub_as(5, 8004, '20.0.0.4', ix2_peering_ips)
emulator.add_as(as5)

as6 = create_stub_as(6, 8005, '30.0.0.4', ix3_peering_ips)
emulator.add_as(as6)

emulator.add_private_peering('10.0.0.1', '20.0.0.1')
emulator.add_private_peering('10.0.0.1', '30.0.0.1')
emulator.add_private_peering('20.0.0.1', '30.0.0.1')

as7 = create_stub_as(7, 8006, '10.0.1.1', [ix1_peering_ips[2]])
emulator.add_as(as7)

as8 = create_stub_as(8, 8007, '20.0.1.1', [ix2_peering_ips[2]])
emulator.add_as(as8)

emulator.add_private_peering('10.0.1.1', '20.0.1.1')

attacker = create_attacker(9, ix1_peering_ips[3])
emulator.add_as(attacker)

emulator.merge_attacker(attacker, {'10.0.0.9': '10.0.0.2/24'})

output_dir = 'output'
emulator.compile(output_dir)
```