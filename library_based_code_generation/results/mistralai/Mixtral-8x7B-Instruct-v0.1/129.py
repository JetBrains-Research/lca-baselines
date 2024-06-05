 ```python
import sys
from scapy.all import IPv6, ICMPv6EchoRequest, sr1, sniff
from impactdecoder import ImpactDecoder

def icmp6_ping(src_ip, dest_ip):
    while True:
        # Create IPv6 packet with source and destination addresses
        ip6 = IPv6(src=src_ip, dst=dest_ip)

        # Create ICMPv6 echo request packet
        icmp = ICMPv6EchoRequest(id=1234, seq=5678)

        # Combine IPv6 and ICMPv6 packets
        packet = ip6 / icmp

        # Send the packet and wait for a reply
        reply = sr1(packet, timeout=5, verbose=0)

        if reply is not None:
            # Decode the reply using ImpactDecoder
            decoded_packet = ImpactDecoder(bytes(reply))

            # Check if the reply is an echo reply
            if decoded_packet.type == 128:
                print(f"Reply size: {len(reply)}")
                print(f"Destination IP: {decoded_packet.src}")
                print(f"Echo sequence number: {decoded_packet.icmp.seq}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python icmp6_ping.py <source_ip> <destination_ip>")
        sys.exit(1)

    src_ip = sys.argv[1]
    dest_ip = sys.argv[2]

    icmp6_ping(src_ip, dest_ip)

if __name__ == "__main__":
    main()
```