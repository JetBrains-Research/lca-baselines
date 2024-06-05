import sys
from scapy.all import *

source_ip = sys.argv[1]
destination_ip = sys.argv[2]

while True:
    packet = IPv6(src=source_ip, dst=destination_ip)/ICMPv6EchoRequest()
    reply = sr1(packet, timeout=1, verbose=0)
    
    if reply:
        if reply.type == 129: # ICMPv6 Echo Reply
            reply.show()
            print("Size of reply: ", len(reply))
            print("Destination IP: ", reply[IPv6].src)
            print("Echo sequence number: ", reply.seq)
    time.sleep(1)