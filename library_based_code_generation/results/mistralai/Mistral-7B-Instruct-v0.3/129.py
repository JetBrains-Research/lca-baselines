 Here is a simple Python script using the Scapy library to perform ICMPv6 ping. Please ensure you have Scapy installed before running the code.

```python
import sys
from scapy.all import IPv6ICMPv6, IPv6, sr1

def ping(src_ip, dest_ip):
    icmp = IPv6ICMPv6(type=8, code=0, id=1, seq=0)
    ip = IPv6(src=src_ip, dst=dest_ip)
    packet = ip / icmp
    reply = sr1(packet, timeout=1, verbose=0)

    if reply:
        if reply.getlayer(IPv6ICMPv6).icmp_type == 0:
            print(f"Reply from {reply.getlayer(IPv6).dst}: Size {len(reply)} Sequence number {reply.getlayer(IPv6ICMPv6).seq}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ping6.py <source_ip> <destination_ip>")
        sys.exit(1)

    source_ip = sys.argv[1]
    destination_ip = sys.argv[2]

    while True:
        ping(source_ip, destination_ip)
```

Save this code as `ping6.py` and run it from the command line with the source and destination IP addresses as arguments:

```
python ping6.py <source_ip> <destination_ip>
```

Replace `<source_ip>` and `<destination_ip>` with the appropriate IPv6 addresses.