```python
import sys
from scapy.all import IPv6, ICMPv6EchoRequest, send, sniff
from scapy.layers.inet6 import ICMPv6EchoReply
from scapy.packet import Raw

def send_icmp6_echo_request(source_ip, destination_ip):
    packet = IPv6(src=source_ip, dst=destination_ip)/ICMPv6EchoRequest()
    send(packet, verbose=0)

def handle_packet(packet):
    if ICMPv6EchoReply in packet:
        print(f"Reply size: {len(packet)}, Destination IP: {packet[IPv6].dst}, Echo Sequence Number: {packet[ICMPv6EchoReply].seq}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_ip> <destination_ip>")
        sys.exit(1)

    source_ip = sys.argv[1]
    destination_ip = sys.argv[2]

    while True:
        send_icmp6_echo_request(source_ip, destination_ip)
        sniff(filter=f"ip6 and src {destination_ip}", prn=handle_packet, count=1, timeout=2)

if __name__ == "__main__":
    main()
```