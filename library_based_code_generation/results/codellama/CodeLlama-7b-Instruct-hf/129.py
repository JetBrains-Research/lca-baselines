  import socket
import struct
import time

def main():
    # Get the source and destination IP addresses from the command line arguments
    src_ip = socket.inet_pton(socket.AF_INET6, sys.argv[1])
    dst_ip = socket.inet_pton(socket.AF_INET6, sys.argv[2])

    # Create an IP6 packet with the source and destination addresses
    ip_packet = struct.pack('!BBHHBB16s16s', 6, 0, 0, 0, 0, 0, src_ip, dst_ip)

    # Create an ICMP echo request packet
    icmp_packet = struct.pack('!BBHH', 128, 128, 0, 0)

    # Send the ICMP echo request to the destination IP
    sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)
    sock.sendto(ip_packet + icmp_packet, (dst_ip, 0))

    # Wait for an echo reply
    while True:
        data, addr = sock.recvfrom(1024)

        # Decode the reply using the ImpactDecoder
        decoder = ImpactDecoder()
        decoded_data = decoder.decode(data)

        # Print the size of the reply, the destination IP, and the echo sequence number
        if decoded_data['type'] == 129:
            print(f'Reply from {addr[0]}')
            print(f'Bytes: {len(data)}')
            print(f'Destination IP: {dst_ip}')
            print(f'Echo sequence number: {decoded_data["seq"]}')

if __name__ == '__main__':
    main()