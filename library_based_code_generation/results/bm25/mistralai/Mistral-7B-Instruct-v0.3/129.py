 Here is a simple Python code using the Scapy library to perform an ICMPv6 ping. Please note that you need to have Scapy installed in your Python environment.

```python
import sys
from scapy.all import IPv6ICMP, IPv6, sr1, Ether

def ping(src_ip, dest_ip):
    icmp = IPv6ICMP(echo_request=1)
    ip = IPv6(src=src_ip, dst=dest_ip)
    packet = ip / icmp
    answer = sr1(packet, timeout=2, verbose=0)

    if answer:
        if answer.haslayer(IPv6ICMP):
            if answer.IPv6ICMP.icmp_type == 128:  # Echo Reply
                print(f"Received reply from {answer.IPv6.dst}: Size {len(answer)} Sequence Number {get_echo_sequence_number(answer)}")

    else:
        print(f"No response from {dest_ip}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ping6.py <source_ip> <destination_ip>")
        sys.exit(1)

    src_ip = sys.argv[1]
    dest_ip = sys.argv[2]

    while True:
        ping(src_ip, dest_ip)
```

This code defines a function `ping` that takes source and destination IP addresses as arguments, creates an ICMPv6 echo request packet, sends it, waits for a reply, and decodes the reply if received. The function then prints the size of the reply, the destination IP, and the echo sequence number if the reply type is an echo reply. The code runs in an infinite loop, sending echo requests and listening for replies.

To run the script, save it as `ping6.py` and execute it from the command line with the source and destination IP addresses as arguments:

```
python ping6.py <source_ip> <destination_ip>
```