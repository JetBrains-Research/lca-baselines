from seed_emulator import PoAServer, TorService, makeEmulatorBaseWith5StubASAndHosts, makeEmulatorBaseWith10StubASAndHosts, TorServer, Service, createDirectory, BotnetClientService, BgpAttackerComponent

# Create ransomware attacker
attacker_as = makeEmulatorBaseWith5StubASAndHosts(1, 1)
attacker_host = attacker_as.hosts[0]
attacker_host.add_service(BgpAttackerComponent())

# Create ransomware victims
victim_as = makeEmulatorBaseWith10StubASAndHosts(2, 16)
for host in victim_as.hosts:
    host.add_service(Service())

# Create Tor nodes
directory_authorities = TorServer("Directory Authorities")
clients = TorServer("Clients")
relays = TorServer("Relays")
exits = TorServer("Exits")
hidden_service = TorServer("Hidden Service")
hidden_service.link_to(attacker_host)

# Create DNS layer
root_server = createDirectory("Root Server")
TLD_server = createDirectory("TLD Server")
ccTLD_server = createDirectory("ccTLD Server")
second_level_zone_servers = [createDirectory(f"Second Level Zone Server {i}") for i in range(5)]
local_dns_server = createDirectory("Local DNS Server")

# Compile emulator using Docker compiler
# Copy necessary files to output directory
# Make script executable