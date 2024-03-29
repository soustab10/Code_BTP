import math
import random
from math import *


class Model:
    def __init__(self, n):
        self.n = 500

        # coordinates of field
        self.x = 500
        self.y = 500
        self.z = -500

        self.x_range = (0, 500)
        self.y_range = (0, 500)
        self.z_range = (-500, 0)
        # Sink Motion pattern
        self.sink_x = self.x * 0.5
        self.sink_y = self.y * 0.5
        self.sink_z = 0
        self.sinkE = 10000  # Energy of sink

        # Optimal Election Probability of a node to become cluster head
        self.p: float = 0.1

        self.tx_range = 100
        self.freq = 10000
        self.data_rate = 35000
        self.f_khz = 10
        self.alpha_f = (
            0.11 * self.f_khz**2 / (1 + self.f_khz**2)
            + 44 * self.f_khz**2 / (4100 + self.f_khz**2)
            + 0.003
            + 2.75e-4 * self.f_khz**2
        )
        # self.num_clusters =
        # %%%%%%%%%%% Energy Model (all values in Joules and each value is for 1byte of data) %%%%%%%%%%%
        # Initial Energy
        self.Eo: float = 2
        self.E_threshold = 0.40 * self.Eo
        # ETX = Energy dissipated in Transmission, ERX = in Receive
        self.Eelec: float = 50 * 0.000000001
        self.ETX: float = 10 * 10e-12
        self.ERX: float = 10 * 10e-12

        # Transmit Amplifier types
        self.Efs: float = 10e-11
        self.Emp: float = 0.0013 * 0.000000000001

        # Data Aggregation Energy
        self.EDA: float = 5 * 0.000000001

        # Computation of do
        self.do: float = sqrt(self.Efs / self.Emp)

        # %%%%%%%%%%%%%%%%%%%%%%%%% Run Time Parameters %%%%%%%%%%%%%%%%%%%%%%%%%
        # maximum number of rounds
        self.rmax = 750
        # Data packet size
        self.data_packet_len = 200

        # Hello packet size
        self.hello_packet_len = 4

        # Number of Packets sent in steady-state phase
        self.NumPacket = 10

        # Radio Range
        self.RR: float = self.tx_range

        self.attn = 10 ** (self.alpha_f / 10)

        self.speed_sound = 1500

        # Layers
        self.num_layers = 5
        self.layer_heights = [0, -80, -170, -270, -380, -500]
        # Fitness function
        self.ff_alpha = 0.5

        # Clusters
        self.num_clusters = 45
        self.clusters_per_layer = int(self.num_clusters / self.num_layers)

        self.num_sinks = 25

        # self.numRx = int(sqrt(self.p * self.n))
        # self.dr = x / self.numRx
        # %%%%%%%%%%%%%%%%%%%%%%%%% END OF PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%


class Sensor:
    def __init__(self):
        self.xd = 0
        self.yd = 0
        self.zd = 0
        self.depth = 0
        self.G = 0
        self.df = 0
        self.type = "N"
        self.E: float = 0
        self.id = 0
        self.dis2sink: float = 0
        self.dis2ch: float = 0
        self.cluster_id = 0  # Member of which Cluster
        self.RR = 0
        self.layer_number = 0
        self.hop_count = 0


def create_sensors(my_model: Model):
    n = my_model.n

    # Configuration sensors
    # created extra one slot for sink
    sensors = [Sensor() for _ in range(n + my_model.num_sinks)]

    # for sink
    """ 
    first n - 1 slots in sensors are for normal sensors. (0 to n-1) 
    nth slot is for sink
    so for n=10, 0-9 are 10 normal sensors and 10th slot is for sink 
    so Sensor[10] = 11th node = sink
    """

    # x_sink = []
    # y_sink = []
    # unf_sink = int(math.sqrt(my_model.num_sinks))
    # print("unf_sink", unf_sink)
    # step_size = my_model.x / unf_sink
    # for i in range(unf_sink):
    #     for j in range(unf_sink):
    #         x_sink.append((i + 1) * step_size)
    #         y_sink.append((j + 1) * step_size)

    x_values = [50, 150, 250, 350, 450]

    # Separate x and y values
    x_sink = []
    y_sink = []

    for x_val in x_values:
        for y_val in x_values:
            x_sink.append(x_val)
            y_sink.append(y_val)
    # x_sink = [250,125, 250, 375, 125, 375, 125, 250, 375]
    # y_sink = [250,125, 125, 125, 250, 250, 375, 375, 375]

    for i in range(0, my_model.num_sinks):
        sensors[n + i].xd = x_sink[i]
        sensors[n + i].yd = y_sink[i]
        sensors[n + i].zd = my_model.sink_z
        sensors[n + i].E = my_model.sinkE
        sensors[n + i].id = my_model.n + i
        sensors[n + i].type = "S"
        sensors[n + i].dis2sink = 0
        sensors[n + i].layer_number = 5
        sensors[n + i].hop_count = 0

    for i, sensor in enumerate(sensors[: -my_model.num_sinks]):

        sensor.xd = random.uniform(my_model.x_range[0], my_model.x_range[1])
        sensor.yd = random.uniform(my_model.y_range[0], my_model.y_range[1])
        sensor.zd = random.uniform(my_model.z_range[0], my_model.z_range[1])

        # Determinate whether in previous periods a node has been cluster-head or not? not=0 and be=n
        sensor.G = 0
        # dead flag. Whether dead or alive S[i].df=0 alive. S[i].df=1 dead.
        sensor.df = 0
        # initially there are not each cluster heads
        sensor.type = "N"
        # initially all nodes have equal Energy
        sensor.E = my_model.Eo
        # id
        sensor.id = i
        # Radio range
        sensor.RR = my_model.RR
        sensor.cluster_id = 0
        # Dist to sink
        sensor.dis2sink = sqrt(
            pow((sensor.xd - sensors[-1].xd), 2) + pow((sensor.yd - sensors[-1].yd), 2)
        )
        sensor.layer_number = get_layer_number(sensor.zd)
        sensor.hop_count = 5 + 1 - sensor.layer_number
        # print(f'Dist to sink: {sensors[-1].id} for {sensor.id} is {sensor.dis2sink}')

    return sensors


def get_layer_number(z):
    layer_heights = [0, -80, -170, -270, -380, -500]
    for i, height in enumerate(layer_heights[::-1], start=0):
        if z <= height:
            return i
    return 0
