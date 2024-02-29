from src.LEACH_create_basics import *


def start(Sensors: list[Sensor], receiver):
    sender = []

    for sensor in Sensors[:-1]:
        if sensor.cluster_id == receiver and sensor.id != receiver:
            sender.append(sensor.id)
            # Todo: UNCOMMENT
            # print(f'sender node: {sensor.id} will send to {receiver} ')

    return sender
