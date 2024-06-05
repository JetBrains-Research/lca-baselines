import sys
from impacket import IP6, ICMP6
from impacket.ImpactDecoder import ImpactDecoder

while True:
    source_ip = sys.argv[1]
    dest_ip = sys.argv[2]

    ip = IP6.IP6()
    ip.set_source_address(source_ip)
    ip.set_destination_address(dest_ip)

    icmp = ICMP6.ICMP6()
    icmp.set_type(ICMP6.ICMP6_ECHO_REQUEST)
    icmp.set_echo_id(1)
    icmp.set_echo_sequence(1)

    ip.contains(icmp)

    data = ip.get_packet()
    # Send the packet and receive the reply
    # Wait for the echo reply
    reply = sendReplyError(data, dest_ip)

    if reply.get_type() == ICMP6.ICMP6_ECHO_REPLY:
        decoder = ImpactDecoder()
        decoded_reply = decoder.decode(reply.get_data())
        print("Reply size: ", len(decoded_reply))
        print("Destination IP: ", decoded_reply.get_destination_address())
        print("Echo sequence number: ", decoded_reply.get_echo_sequence_number())