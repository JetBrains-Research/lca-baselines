 import sys
from scapy.all import *

def icmp6\_ping(src\_ip, dest\_ip):
while True:
ip = IPv6(src=src\_ip, dst=dest\_ip)
icmp = ICMP6EchoRequest(data=RandShort())
packet = ip/icmp
response = sr1(packet, timeout=2, verbose=0)

if response is not None:
if response.haslayer(ICMP6EchoReply):
print(f"Received echo reply from {dest\_ip}, size: {len(response[ICMP6EchoReply].data)}")
else:
print(f"Received unexpected response from {dest\_ip}: {response.summary()}")

if __name__ == "__main__":
if len(sys.argv) != 3:
print("Usage: python icmp6\_ping.py <source\_ip> <destination\_ip>")
sys.exit(1)

src\_ip = sys.argv[1]
dest\_ip = sys.argv[2]

icmp6\_ping(src\_ip, dest\_ip)