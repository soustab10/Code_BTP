from src.LEACH_create_basics import *


def start(Sensors: list[Sensor], receiver):
    sender = []

    for sensor in Sensors[:-1]:
        if sensor.cluster_id == receiver.cluster_id and sensor.id != receiver.id:
            sender.append(sensor)
            # Todo: UNCOMMENT
            # print(f'sender node: {sensor.id} will send to {receiver} ')

    return sender
