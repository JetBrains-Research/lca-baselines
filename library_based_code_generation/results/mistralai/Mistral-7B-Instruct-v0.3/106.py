 Here is a simplified Python code using the seed-emulator library that follows your instructions. Please note that this is a basic example and might need adjustments to fit your specific needs. Also, the Docker part assumes you have a Dockerfile in the same directory.

```python
import seed_emulator as se

def create_stub_as(as_id, web_port=80, router_port=179):
    web_server = se.WebServer(port=web_port)
    router = se.Router(as_id, web_server)
    return router

def create_internet_exchange(ixp_id):
    ixp = se.InternetExchange(ixp_id)
    return ixp

def create_autonomous_system(as_id, ixp, routers):
    as_system = se.AutonomousSystem(as_id)
    as_system.add_router(routers[0])
    for ixp_peer in ixps:
        as_system.add_private_peering(ixp_peers[ixp_peer])
    return as_system

def create_routers_and_ixps():
    routers = [create_stub_as(as_id) for as_id in range(1, 6)]
    ixps = [create_internet_exchange(ixp_id) for ixp_id in range(1, 4)]
    ixp_peers = {ixp: routers for ixp in ixps}
    as_systems = [create_autonomous_system(as_id, ixp, routers) for as_id, (ixp, routers) in enumerate(zip(ixps, routers))]
    return routers, ixps, as_systems, ixp_peers

def create_bgp_attacker(as_id, prefixes_to_hijack):
    bgp_attacker = se.BGPAttacker(as_id)
    for prefix in prefixes_to_hijack:
        bgp_attacker.hijack_prefix(prefix)
    return bgp_attacker

def merge_bgp_attacker(emulator, bgp_attacker):
    emulator.add_component(bgp_attacker)
    return emulator

def render_and_dockerize(emulator, output_dir):
    emulator.render()
    docker_cmd = f"docker build -t seed-emulator . && docker run -d -p 8080:80 seed-emulator"
    subprocess.run(docker_cmd, shell=True, cwd=output_dir)

routers, ixps, as_systems, ixp_peers = create_routers_and_ixps()
bgp_attacker = create_bgp_attacker(5, [prefix1, prefix2])
emulator = se.Emulator()
emulator.add_component(bgp_attacker)
for as_system in as_systems:
    emulator.add_component(as_system)
for ixp, routers in ixp_peers.items():
    emulator.add_component(ixp)
    for router in routers:
        ixp.add_peer(router)
emulator = merge_bgp_attacker(emulator, bgp_attacker)
render_and_dockerize(emulator, "output_dir")
```

This code creates three internet exchanges, multiple stub autonomous systems that join these exchanges, two autonomous systems with routers that join different networks and internet exchanges, private peerings between different autonomous systems, and a BGP attacker that hijacks certain prefixes and joins an internet exchange. The new emulator is then merged with the BGP attacker and rendered, and finally, it is dockerized.