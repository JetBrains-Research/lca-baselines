```python
import sys
from scapy.all import IPv6, ICMPv6EchoRequest, send, sniff
from scapy.layers.inet6 import icmp6types

def send_icmp6_echo_request(source_ip, destination_ip):
    packet = IPv6(src=source_ip, dst=destination_ip)/ICMPv6EchoRequest()
    send(packet)

def handle_echo_reply(packet):
    if packet.type == icmp6types.ICMPv6EchoReply:
        print(f"Reply size: {len(packet)}, Destination IP: {packet[IPv6].dst}, Echo sequence number: {packet.seq}")

def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <source_ip> <destination_ip>")
        sys.exit(1)

    source_ip = sys.argv[1]
    destination_ip = sys.argv[2]

    while True:
        send_icmp6_echo_request(source_ip, destination_ip)
        sniff(filter=f"ip6 and src {destination_ip}", prn=handle_echo_reply, count=1, timeout=5)

if __name__ == "__main__":
    main()
```