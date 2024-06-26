from src.LEACH_create_basics import *


def send_rec(sensors: list[Sensor], sender, sap):
    # for senders
    # the packet will be sent only if the sender has energy left after transmission
    if sensors[sender].E > 0:
        # Send a packet and increment counter by 1
        sap += 1
        # print(f'{sender} sent packet. New energy of {sender} = {sensors[sender].E}')
    else:
        # print(f"node {sensors[sender]} is Dead! :( look how they massacpink my node.")
        sensors[sender].df = 1

    return sap


def start(sensors: list[Sensor], my_model: Model, senders: list, receivers: list, srp, rrp, sdp, rdp, packet_type: str):
    sent_packets = 0  # Number of sent packets
    rec_packets = 0  # Number of received packets

    PacketSize = my_model.hello_packet_len if packet_type == 'Hello' else my_model.data_packet_len

    # todo: do not increment rdp and/or sdp if one of them is dead

    # Energy dissipated from Sensors for Sending a packet
    # Each sender will send to each receiver
    for sender in senders:
        for receiver in receivers:
            # print("########sender is ", sender.id, "and rec is ", receiver.id)
            # print()
            distance = sqrt(
                pow(sensors[sender.id].xd - sensors[receiver.id].xd, 2) +
                pow(sensors[sender.id].yd - sensors[receiver.id].yd, 2)
            )
            # print(f"dist b/w sender: {sender.id} and receiver: {receiver.id} is: {distance}")

            if distance > my_model.do:
                sensors[sender.id].E -= (my_model.ETX * PacketSize + my_model.Emp * PacketSize * pow(distance, 4))*my_model.NumPacket
                sent_packets = send_rec(sensors, sender.id, sent_packets)

            else:
                sensors[sender.id].E -= (my_model.ETX * PacketSize + my_model.Efs * PacketSize * pow(distance, 2))*my_model.NumPacket
                sent_packets = send_rec(sensors, sender.id, sent_packets)

    for receiver in receivers:
        sensors[receiver.id].E -= ((my_model.ERX + my_model.EDA) * PacketSize)*my_model.NumPacket

    # for receivers
    # Energy dissipated from receivers for Receiving a packet, if the sender died during transmission,
    # the energy of receiver will be wasted but it will not receive any packet
    for sender in senders:
        for receiver in receivers:
            if sensors[receiver.id].E > 0 and sensors[sender.id].E > 0:
                # Received a Packet
                rec_packets += 1
                # print(f'{receiver.id} received a packet, new energy of {receiver.id} = {sensors[receiver.id].E}')
            elif sensors[receiver.id].E < 0:
                sensors[receiver.id].df = 1

    if packet_type == 'Hello':
        srp += sent_packets
        rrp += rec_packets
        # print(f"incremented srp by {sent_packets} and rrp by {rec_packets}")
    elif packet_type == 'Data':
        sdp += sent_packets
        rdp += rec_packets
        # print(f"incremented sdp by {sent_packets} and rdp by {rec_packets}")

    # print()
    return srp, rrp, sdp, rdp
