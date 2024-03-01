from math import inf

from src.LEACH_create_basics import *


def start(Sensors: list[Sensor], my_model: Model, round_number):
    

    srp = 0  # counter number of sent routing packets
    rrp = 0  # counter number of receive routing packets
    sdp = 0  # counter number of sent data packets to sink
    rdp = 0  # counter number of receive data packets by sink

    return srp, rrp, sdp, rdp
