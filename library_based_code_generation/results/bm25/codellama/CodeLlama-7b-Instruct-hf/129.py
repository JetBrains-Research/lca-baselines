  import socket
import struct
import time

def main(source_ip, destination_ip):
    # Create an IPv6 socket
    sock = socket.socket(socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6)

    # Set the source and destination IP addresses
    sock.bind((source_ip, 0))
    sock.connect((destination_ip, 0))

    # Create an ICMP echo request packet
    echo_request = struct.pack("!BBHHH", 128, 129, 0, 0, 0)

    # Send the echo request
    sock.send(echo_request)

    # Wait for an echo reply
    while True:
        data = sock.recv(1024)
        if not data:
            break

        # Decode the reply using the ImpactDecoder
        decoder = ImpactDecoder()
        decoder.decode(data)

        # Check if the reply type is an echo reply
        if decoder.get_type() == 129:
            # Print the size of the reply, the destination IP, and the echo sequence number
            print("Reply size:", decoder.get_size())
            print("Destination IP:", decoder.get_destination_ip())
            print("Echo sequence number:", decoder.get_echo_sequence_number())

        # Send another echo request
        sock.send(echo_request)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])