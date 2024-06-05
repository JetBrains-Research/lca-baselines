import seed_emulator

# Create ransomware service
ransomware_attacker = seed_emulator.RansomwareAttacker(host='autonomous_system', botnet=False, tor=False)
ransomware_victims = [seed_emulator.RansomwareVictim(host='host', botnet=False) for _ in range(16)]

# Create Tor service
directory_authorities = [seed_emulator.TorNode(node_type='directory_authority') for _ in range(3)]
clients = [seed_emulator.TorNode(node_type='client') for _ in range(5)]
relays = [seed_emulator.TorNode(node_type='relay') for _ in range(5)]
exits = [seed_emulator.TorNode(node_type='exit') for _ in range(3)]
hidden_service = seed_emulator.TorNode(node_type='hidden_service')
hidden_service.link_to(ransomware_attacker)

# Create DNS layer
root_server = seed_emulator.DNSServer(server_type='root')
TLD_servers = [seed_emulator.DNSServer(server_type='TLD') for _ in range(3)]
ccTLD_servers = [seed_emulator.DNSServer(server_type='ccTLD') for _ in range(5)]
second_level_zone_servers = [seed_emulator.DNSServer(server_type='second_level_zone') for _ in range(10)]
local_DNS_server = seed_emulator.DNSServer(server_type='local')

# Compile emulator
emulator = seed_emulator.Emulator()
emulator.compile_with_docker(victim_base_image='custom_victim_image', attacker_base_image='custom_attacker_image')
emulator.copy_files_to_output_directory()
emulator.make_script_executable()