import pprint
from math import *
import matplotlib
import time

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"

from src import LEACH_create_basics
from src import LEACH_select_ch
from src import findReceiver
from src import find_sender
from src import join_to_nearest_ch
from src import reset_sensors
from src import send_receive_packets
from src import LEACH_plotter


def zeros(row, column):
    re_list = []
    for x in range(row):
        temp_list = [0 for _ in range(column)]
        if row == 1:
            re_list.extend(temp_list)
        else:
            re_list.append(temp_list)

    return re_list


avg_e2edelay_md = []


class LEACHSimulation:

    def __init__(self, n=500):
        self.n = n  # Number of Nodes in the field

        self.dead_num = 0  # Number of dead nodes
        self.no_of_ch = 0  # counter for CHs
        self.flag_first_dead = 0  # flag_first_dead
        self.initEnergy = 0  # Initial Energy

        self.my_model = LEACH_create_basics.Model(self.n)  #
        self.SRP = zeros(1, self.my_model.rmax + 1)  # number of sent routing packets
        self.RRP = zeros(1, self.my_model.rmax + 1)  # number of receive routing packets
        self.SDP = zeros(1, self.my_model.rmax + 1)  # number of sent data packets
        self.RDP = zeros(1, self.my_model.rmax + 1)  # number of receive data packets
        self.avg_e2edelay = zeros(
            1, self.my_model.rmax + 1
        )  # number of receive data packets
        self.pkt_count = zeros(1, self.my_model.rmax + 1)
        self.rcd_sink = zeros(
            1, self.my_model.rmax + 1
        )  # received data packets by sink
        self.layer_heights = [0, -80, -170, -270, -380, -500]
        # counter for bit transmitted to Bases Station and Cluster Heads
        self.srp = 0  # counter number of sent routing packets
        self.rrp = 0  # counter number of receive routing packets
        self.sdp = 0  # counter number of sent data packets to sink
        self.rdp = 0  # counter number of receive data packets by sink

        # ########################################################
        # ############# For initialization_main_loop #############
        # ########################################################
        self.dead_num = []
        self.packets_to_base_station = 0
        self.first_dead_in = -1
        self.list_CH = []
        # This section Operate for each epoch
        # self.member = []  # Member of each cluster in per period      # Not used

        self.alive = self.n

        # self.total_energy_dissipated = zeros(1, self.myModel.rmax + 1)
        # self.AllSensorEnergy = zeros(1, self.myModel.rmax + 1)
        self.sum_dead_nodes = zeros(1, self.my_model.rmax + 1)
        self.ch_per_round = zeros(1, self.my_model.rmax + 1)

        # all sensors should be alive in start
        self.alive_sensors = zeros(1, self.my_model.rmax + 1)
        self.alive_sensors[0] = self.n

        self.sum_energy_left_all_nodes = zeros(1, self.my_model.rmax + 1)
        self.avg_energy_All_sensor = zeros(1, self.my_model.rmax + 1)
        self.consumed_energy = zeros(1, self.my_model.rmax + 1)
        self.Enheraf = zeros(1, self.my_model.rmax + 1)

        self.cluster_centers = []
        ##############################################
        # todo: test
        ##############################################
        print("self.my_model")
        print(vars(self.my_model))
        print("length of below 4=", len(self.SRP))
        print("self.SRP", self.SRP)
        print("self.RRP", self.RRP)
        print("self.SDP", self.SDP)
        print("self.RDP", self.RDP)
        print("----------------------------------------------")

    def start(self):
        print("#################################")
        print("############# Start #############")
        print("#################################")
        print()

        self.__create_sen()

        self.__print_sensors()
        # self.__plot_sensors()
        self.__start_simulation()

        # self.__plot_clusters()
        self.__main_loop()
        # self.__statistics(round_number=self.my_model.rmax)
        # Todo: all plotting should be done in Leach_plotter file

        LEACH_plotter.start(self)
        self.avg_e2edelay[0] = self.avg_e2edelay[1]
        plt.xlim(left=0, right=self.my_model.rmax)
        plt.ylim(bottom=0, top=1.5)
        plt.plot(self.avg_e2edelay, color="olive")
        plt.title(
            "Average End-to-End Delay (Per Round)",
        )
        plt.xlabel(
            "Rounds",
        )
        plt.ylabel(
            "Delay (s)",
        )
        plt.show()

        print("-------------------- XXX --------------------")
        print("############# END of simulation #############")
        print("-------------------- XXX --------------------")

    def __check_dead_num(self, round_number):
        # if sensor is dead
        for sensor in self.Sensors:
            if sensor.E <= 0 and sensor not in self.dead_num:
                sensor.df = 1
                self.dead_num.append(sensor)

                print(f"{sensor.id} is dead, \ndeadnum=")
                for _ in self.dead_num:
                    print(_.id, end=" ")
                print()

        # flag it as dead
        if len(self.dead_num) > 0 and self.flag_first_dead == 0:
            # Save the period in which the first node died
            self.first_dead_in = round_number
            self.flag_first_dead = 1

            # print(f"First dead in round: {round_number}")

    def __create_sen(self):
        print("##########################################")
        print("############# Create Sensors #############")
        print("##########################################")
        print()

        # Create a random scenario & Load sensor Location
        # configure sensors
        self.Sensors = LEACH_create_basics.create_sensors(self.my_model)

        for sensor in self.Sensors[: self.my_model.num_sinks]:
            self.initEnergy += sensor.E

        # We will have full energy in start
        self.sum_energy_left_all_nodes[0] = self.initEnergy
        self.avg_energy_All_sensor[0] = self.initEnergy / self.n

        # todo: test
        print(self.Sensors)
        # print("self.initEnergy", self.initEnergy)
        # print("----------------------------------------------")

    def __plot_sensors(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x_vals = [node.xd for node in self.Sensors if node.type == "N"]
        y_vals = [node.yd for node in self.Sensors if node.type == "N"]
        z_vals = [node.zd for node in self.Sensors if node.type == "N"]
        x_vals_ch = [node.xd for node in self.Sensors if node.type == "C"]
        y_vals_ch = [node.yd for node in self.Sensors if node.type == "C"]
        z_vals_ch = [node.zd for node in self.Sensors if node.type == "C"]

        sink_x_vals = [sensor.xd for sensor in self.Sensors if sensor.type == "S"]
        sink_y_vals = [sensor.yd for sensor in self.Sensors if sensor.type == "S"]
        sink_z_vals = [sensor.zd for sensor in self.Sensors if sensor.type == "S"]

        x_vals = x_vals[: -self.my_model.num_sinks]
        y_vals = y_vals[: -self.my_model.num_sinks]
        z_vals = z_vals[: -self.my_model.num_sinks]

        ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o", label="Nodes")
        # ax.scatter(x_vals_ch, y_vals_ch, z_vals_ch, c="olive", marker="x", label="CHs")

        # Plot olive planes at specified heights
        for height in self.layer_heights:
            x_plane = np.linspace(min(x_vals), max(x_vals), 100)
            y_plane = np.linspace(min(y_vals), max(y_vals), 100)
            x_plane, y_plane = np.meshgrid(x_plane, y_plane)
            z_plane = np.full_like(x_plane, height)
            ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.1, color="olive")

        ax.scatter(
            sink_x_vals,
            sink_y_vals,
            sink_z_vals,
            c="olive",
            marker="^",
            label="Sink Nodes",
        )

        ax.set_xlabel(
            "X-axis",
        )
        ax.set_ylabel(
            "Y-axis",
        )
        ax.set_zlabel(
            "Z-axis",
        )
        ax.set_title(
            "3D UWSN",
        )

        plt.legend()
        plt.show()

    def __print_sensors(self):
        # print("##########################################")
        # print("############# # print Sensors #############")
        # print("##########################################")
        # print()
        ct = 0
        num_clusters = set()
        for sensor in self.Sensors:
            print(
                sensor.id,
                sensor.xd,
                sensor.yd,
                sensor.zd,
                sensor.layer_number,
                sensor.cluster_id,
                sensor.type,
            )
            num_clusters.add(sensor.cluster_id)
            if sensor.type == "C":
                ct += 1

        # print("Total Number of CHs:", ct)
        # print("Total Number of Clusters:", len(num_clusters))
        # print("Cluster centers:", self.cluster_centers)
        # print("----------------------------------------------")

    def __plot_clusters(self, layer_number=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        cluster_sensors = {}
        ch_coordinates = {"x": [], "y": [], "z": []}

        for sensor in self.Sensors:
            if sensor.cluster_id not in cluster_sensors:
                cluster_sensors[sensor.cluster_id] = {"x": [], "y": [], "z": []}

            cluster_sensors[sensor.cluster_id]["x"].append(sensor.xd)
            cluster_sensors[sensor.cluster_id]["y"].append(sensor.yd)
            cluster_sensors[sensor.cluster_id]["z"].append(sensor.zd)

            if sensor.type == "C":
                ch_coordinates["x"].append(sensor.xd)
                ch_coordinates["y"].append(sensor.yd)
                ch_coordinates["z"].append(sensor.zd)

        for cluster_id, coordinates in cluster_sensors.items():
            ax.scatter(
                coordinates["x"],
                coordinates["y"],
                coordinates["z"],
                label=f"Cluster {cluster_id}",
            )

        ax.scatter(
            ch_coordinates["x"],
            ch_coordinates["y"],
            ch_coordinates["z"],
            c="black",
            marker="x",
            s=100,
            label="Cluster Heads",
        )

        ax.set_xlabel(
            "X-axis",
        )
        ax.set_ylabel(
            "Y-axis",
        )
        ax.set_zlabel(
            "Z-axis",
        )
        ax.set_title(
            "3D Sensor Network by Cluster with Cluster Heads",
        )
        # plt.legend()
        plt.show()

    def __start_simulation(self):
        print("############################################")
        print("############# Start Simulation #############")
        print("############################################")
        print()

        print("#######################################################################")
        print("############# Sink broadcast 'Hello' message to all nodes #############")
        print("#######################################################################")
        print()

        self.sender = self.Sensors[-self.my_model.num_sinks :]
        # List of senders, for start_sim, sink will send hello packet to all
        self.receivers = [
            self.Sensors[i] for i in range(self.n)
        ]  # List of senders, for start_sim, All nodes will receive from sink

        # todo: test
        # print("Senders: ", end="")
        # print(self.sender)
        # print("Receivers: ", end="")
        # print(self.receivers)
        # print()

        self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
            self.Sensors,
            self.my_model,
            self.sender,
            self.receivers,
            self.srp,
            self.rrp,
            self.sdp,
            self.rdp,
            packet_type="Hello",
        )

        # todo: test
        # print("self.srp", self.srp)
        # print("self.rrp", self.rrp)
        # print("self.sdp", self.sdp)
        # print("self.rdp", self.rdp)
        # print("Sensors: \n", print(self.Sensors))

        # Save metrics, Round 0 is initialization phase where all nodes send routing packets (hello) to Sink as above
        self.SRP[0] = self.srp
        self.RRP[0] = self.rrp
        self.SDP[0] = self.sdp
        self.RDP[0] = self.rdp

        # todo: test
        # print("self.SRP", self.SRP)
        # print("self.RRP", self.RRP)
        # print("self.SDP", self.SDP)
        # print("self.RDP", self.RDP)

    def __main_loop(self):
        print("#############################################")
        print("############# Main loop program #############")
        print("#############################################")
        print()

        for round_number in range(1, self.my_model.rmax + 1):
            self.r = round_number
            print("#####################################")
            print(f"############# Round {round_number} #############")
            print("#####################################")

            self.srp, self.rrp, self.sdp, self.rdp = reset_sensors.start(
                self.Sensors, self.my_model, round_number
            )

            # todo: test

            # print(self.Sensors)

            # ############# cluster head election #############

            self.__cluster_head_selection_phase(round_number)
            self.no_of_ch = len(self.list_CH)  # Number of CH in per period

            self.__steady_state_phase_2(round_number)
            # self.__steady_state_phase(round_number)

            # # if sensor is dead
            self.__check_dead_num(round_number)

            self.__statistics(round_number)
            if self.pkt_count[round_number] != 0:
                self.avg_e2edelay[round_number] = (
                    self.avg_e2edelay[round_number] / self.pkt_count[round_number]
                )
            # if all nodes are dead or only sink is left, exit
            if len(self.dead_num) >= self.n:
                self.lastPeriod = round_number
                print(
                    f"All nodes are dead. (dead={len(self.dead_num)}) in round {round_number}"
                )
                break

    def __cluster_head_selection_phase(self, round_number):
        print("#################################################")
        print("############# cluster head election #############")
        print("#################################################")
        print()
        # print("Clusters of Current Round:",self.list_CH)
        # Selection Candidate Cluster Head Based on LEACH Set-up Phase
        # ct=1
        # for sensor in self.Sensors[:-self.my_model.num_sinks]:
        # ct=1
        # for sensor in self.Sensors[:-self.my_model.num_sinks]:
        #         sensor.type = 'C'
        #         self.list_CH.append(sensor)
        #         sensor.cluster_id=ct
        #         ct+=1

        # for sensor in self.Sensors[:-self.my_model.num_sinks]:
        #     sensor.type = 'N'

        # self.list_CH = []
        # self.perform_clustering(self.my_model.clusters_per_layer)

        # if round_number <= 1:
        #     self.perform_clustering()
        # else:
        #     cluster_ids_newch = []
        #     for cluster_head in self.list_CH:
        #         if (
        #             cluster_head.E <= self.my_model.E_threshold
        #             and cluster_head.type != "S"
        #         ):
        #             cluster_head.type = "N"
        #             cluster_ids_newch.append(cluster_head.cluster_id)

        #     self.list_CH = [
        #         ch for ch in self.list_CH if ch.cluster_id not in cluster_ids_newch
        #     ]
        #     print(
        #         "Cluster Head Relected ***************************************",
        #         len(cluster_ids_newch),
        #     )
        #     for c_id in cluster_ids_newch:
        #         self.perform_cluster_id_new_ch_2(c_id)

        # join_to_nearest_ch.start(self.Sensors, self.my_model, self.list_CH)
        # self.__print_sensors()
        # self.__plot_clusters()

    def __steady_state_phase_2(self, round_number):
        # print("##############################################")
        # print("############# steady state phase #############")
        # print("##############################################")
        # print()

        # for each sensor -> send data to its CH ->  CH send data to upper layer/hop count -> If CH_hop count = 1 send data to sink

        for sensor in self.Sensors:
            current_time = time.time()
            distance_route = 0.00
            node_route = 0
            sender = None
            receiver = None
            if sensor.type == "N":
                # receiver = [
                #         node
                #         for node in self.Sensors
                #         if node.cluster_id == sensor.cluster_id and node.type == "C"
                #     ][0]
                # sender = sensor
                # distance_route += sqrt(
                #         pow((sender.xd - receiver.xd), 2)
                #         + pow((sender.yd - receiver.yd), 2)
                #         + pow((sender.zd - receiver.zd), 2)
                #     )
                # node_route += 1
                # self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
                #         self.Sensors,
                #         self.my_model,
                #         [sender],
                #         [receiver],
                #         self.srp,
                #         self.rrp,
                #         self.sdp,
                #         self.rdp,
                #         packet_type="Data",
                #     )

                is_sent_to_sink = False
                sender = sensor  # CH is the new sender now
                while is_sent_to_sink != True:

                    # nearest_receiver = self.find_valid_receiver_5(sender)
                    # nearest_receiver = self.find_nearest_sink(sender)
                    nearest_receiver = self.find_valid_receiver_7(sender)
                    self.rcd_sink[round_number] += 1

                    # print("Nearest receiver:", nearest_receiver)
                    # print("Id:",receiver.id)
                    if nearest_receiver == None:
                        nearest_receiver = self.find_nearest_sink(sender)

                    distance_route += sqrt(
                        pow((sender.xd - nearest_receiver.xd), 2)
                        + pow((sender.yd - nearest_receiver.yd), 2)
                        + pow((sender.zd - nearest_receiver.zd), 2)
                    )
                    node_route += 1

                    if nearest_receiver.hop_count <= 1:
                        is_sent_to_sink = True
                        self.receivers = [self.Sensors[self.n :]]  # Sink

                        # Assuming you have a reference point (x_ref, y_ref, z_ref)
                        x_ref, y_ref, z_ref = (
                            sender.xd,
                            sender.yd,
                            sender.zd,
                        )  # Sender coordinates

                        # Calculate distances from the reference point to all sensors except the sender
                        distances = [
                            (
                                (sensor.xd - x_ref) ** 2
                                + (sensor.yd - y_ref) ** 2
                                + (sensor.zd - z_ref) ** 2
                            )
                            ** 0.5
                            for sensor in self.Sensors[
                                self.n : self.n + self.my_model.num_sinks
                            ]
                        ]

                        # Find the index of the sensor with the smallest distance (nearest receiver)
                        nearest_receiver_index = distances.index(min(distances))

                        # Assign the nearest receiver to self.receivers
                        self.receivers = [self.Sensors[nearest_receiver_index]]
                        sender = [sender]

                        # print(f"node {sender} will send directly to sink ")
                        self.srp, self.rrp, self.sdp, self.rdp = (
                            send_receive_packets.start(
                                self.Sensors,
                                self.my_model,
                                sender,
                                self.receivers,
                                self.srp,
                                self.rrp,
                                self.sdp,
                                self.rdp,
                                packet_type="Data",
                            )
                        )

                        distance_route += min(distances)
                        node_route += 1
                        self.avg_e2edelay[round_number] += (
                            distance_route / self.my_model.speed_sound
                            + 0.1 * node_route
                        )
                        self.pkt_count[round_number] += 1

                    else:
                        self.srp, self.rrp, self.sdp, self.rdp = (
                            send_receive_packets.start(
                                self.Sensors,
                                self.my_model,
                                [sender],
                                [nearest_receiver],
                                self.srp,
                                self.rrp,
                                self.sdp,
                                self.rdp,
                                packet_type="Data",
                            )
                        )

                    sender = nearest_receiver

    def __steady_state_phase(self, round_number):
        print("##############################################")
        print("############# steady state phase #############")
        print("##############################################")
        print()

        for i in range(
            self.my_model.NumPacket
        ):  # Number of Packets to be sent in steady-state phase

            print("#############################################################")
            print("############# All sensor send data packet to CH #############")
            print("#############################################################")
            print()

            for receiver in self.list_CH:
                sender = find_sender.start(self.Sensors, receiver)

                # todo: test
                print("sender: ", sender)
                print("receiver: ", receiver)
                print()

                self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
                    self.Sensors,
                    self.my_model,
                    sender,
                    [receiver],
                    self.srp,
                    self.rrp,
                    self.sdp,
                    self.rdp,
                    packet_type="Data",
                )

                # todo: test
                print("self.srp", self.srp)
                print("self.rrp", self.rrp)
                print("self.sdp", self.sdp)
                print("self.rdp", self.rdp)
                print(
                    "Sensors: ",
                )
                print(self.Sensors)
                print()

        # ####################################################################################################
        # ############# send Data packet directly from nodes(that aren't in any cluster) to Sink #############
        # ####################################################################################################

        for sender in self.Sensors:
            # if the node has sink as its CH but it's not sink itself and the node is not dead
            if sender.cluster_id == self.n and sender.id != self.n and sender.E > 0:
                self.receivers = [self.Sensors[self.n]]  # Sink
                sender = [sender]

                print(f"node {sender} will send directly to sink ")
                self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
                    self.Sensors,
                    self.my_model,
                    sender,
                    self.receivers,
                    self.srp,
                    self.rrp,
                    self.sdp,
                    self.rdp,
                    packet_type="Data",
                )

        # ###################################################################################
        # ############# Send Data packet from CH to Sink after Data aggregation #############
        # ###################################################################################

        print("senders (or CH) = ", self.list_CH)

        for sender in self.list_CH:
            self.receivers = [self.Sensors[self.n]]  # Sink

            # todo: test
            print("sender: ", sender)
            print("receiver: ", self.receivers)

            self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
                self.Sensors,
                self.my_model,
                [sender],
                self.receivers,
                self.srp,
                self.rrp,
                self.sdp,
                self.rdp,
                packet_type="Data",
            )

            # todo: test
            print("self.srp", self.srp)
            print("self.rrp", self.rrp)
            print("self.sdp", self.sdp)
            print("self.rdp", self.rdp)
            print(
                "Sensors: ",
            )
            print(self.Sensors)
            print()

    def __statistics(self, round_number):
        print("######################################")
        print("############# STATISTICS #############")
        print("######################################")

        self.sum_dead_nodes[round_number] = len(self.dead_num)
        self.ch_per_round[round_number] = self.no_of_ch
        self.SRP[round_number] = self.srp
        self.RRP[round_number] = self.rrp
        self.SDP[round_number] = self.sdp
        self.RDP[round_number] = self.rdp

        self.alive = 0
        sum_energy_left_all_nodes_in_curr_round = 0
        for sensor in self.Sensors[: -self.my_model.num_sinks]:
            if sensor.E > 0:
                self.alive += 1
                sum_energy_left_all_nodes_in_curr_round += sensor.E

        self.alive_sensors[round_number] = self.alive
        self.sum_energy_left_all_nodes[round_number] = (
            sum_energy_left_all_nodes_in_curr_round
        )
        if self.alive:
            self.avg_energy_All_sensor[round_number] = (
                sum_energy_left_all_nodes_in_curr_round / self.alive
            )
        else:
            self.avg_energy_All_sensor[round_number] = 0
        self.consumed_energy[round_number] = (
            self.initEnergy - self.sum_energy_left_all_nodes[round_number]
        ) / self.n

        En = 0
        for sensor in self.Sensors:
            if sensor.E > 0:
                En += pow(sensor.E - self.avg_energy_All_sensor[round_number], 2)

        if self.alive:
            self.Enheraf[round_number] = En / self.alive
        else:
            self.Enheraf[round_number] = 0

        # todo: test
        # print("round number:", round_number)
        # print("len(self.SRP)", len(self.SRP))
        # print("self.SRP", self.SRP)
        # print("self.RRP", self.RRP)
        # print("self.SDP", self.SDP)
        # print("self.RDP", self.RDP)
        # print("----------------------------------------------")

        # print('self.total_energy_dissipated', self.total_energy_dissipated)
        # print('self.AllSensorEnergy', self.AllSensorEnergy)
        # print("self.sum_dead_nodes", self.sum_dead_nodes)
        # print("self.ch_per_round", self.ch_per_round)
        # print("self.alive_sensors", self.alive_sensors)
        # print("self.sum_energy_all_nodes", self.sum_energy_left_all_nodes)
        # print("self.avg_energy_All_sensor", self.avg_energy_All_sensor)
        # print("self.consumed_energy", self.consumed_energy)
        # print("self.Enheraf", self.Enheraf)

        filename = "sim_data.txt"

        with open(filename, "a") as file:
            file.write(f"Round number: {round_number}\n")
            file.write(f"Length of SRP: {len(self.SRP)}\n")
            file.write(f"SRP: {self.SRP}\n")
            file.write(f"RRP: {self.RRP}\n")
            file.write(f"SDP: {self.SDP}\n")
            file.write(f"RDP: {self.RDP}\n")
            file.write("----------------------------------------------\n")
            file.write(f"Sum of dead nodes: {self.sum_dead_nodes[round_number]}\n")
            file.write(f"CH per round: {self.ch_per_round[round_number]}\n")
            file.write(f"Alive sensors: {self.alive_sensors[round_number]}\n")
            file.write(
                f"Sum energy left in all nodes: {self.sum_energy_left_all_nodes[round_number]}\n"
            )
            file.write(
                f"Average energy of all sensors: {self.avg_energy_All_sensor[round_number]}\n"
            )
            file.write(f"Consumed energy: {self.consumed_energy[round_number]}\n")
            file.write(f"Enheraf: {self.Enheraf[round_number]}\n")
            file.write(f"Rcd at Sink: {self.rcd_sink[round_number]}\n\n")
            file.write(f"Avg E2E Delay: {self.avg_e2edelay}\n")
            avg_e2edelay_md.append(self.avg_e2edelay[round_number])
            file.write(f"Packets Received: {self.pkt_count[round_number]}\n")
            file.write(f"Energy of all nodes:" + "\n")
            for sensor in self.Sensors:
                file.write(f" {sensor.id}: {sensor.E}, ")
            file.write("\n\n")

        # print("----------------------------------------------")

    def perform_clustering(self):
        total_depth = abs(self.my_model.z_range[0] - self.my_model.z_range[1])

        # Iterate through layers

        for layer in range(1, self.my_model.num_layers + 1):
            layer_nodes = [node for node in self.Sensors if node.layer_number == layer]
            # print(layer_nodes)
            if not layer_nodes:
                continue  # Skip layers with no nodes

            # Combine node coordinates into a single array for clustering
            node_coordinates = [[node.xd, node.yd, node.zd] for node in layer_nodes]
            k = len(layer_nodes) // self.my_model.clusters_per_layer
            # print(node_coordinates)
            # Use KMeans clustering with explicit n_init
            kmeans = KMeans(
                n_clusters=k, n_init=int(len(layer_nodes) / k), random_state=0
            )  # Set n_init explicitly
            kmeans.fit(node_coordinates)
            # print(kmeans.labels_)
            # Assign each node to its respective cluster

            for i, node in enumerate(layer_nodes):
                node.cluster_id = kmeans.labels_[i]

            # Store the cluster centers
            self.cluster_centers = kmeans.cluster_centers_

            # Select cluster heads based on a fitness function

            self.select_cluster_heads(layer_nodes, layer)

    def perform_clustering_main(self):
        cluster_id = 1
        block_size = self.my_model.x / sqrt(
            self.my_model.clusters_per_layer
        )  # Calculate size of each block
        itr = int(np.sqrt(self.my_model.clusters_per_layer))
        for layer in range(1, self.my_model.num_layers + 1):
            for row in range(itr):
                for col in range(itr):
                    x_start = col * block_size
                    x_end = (col + 1) * block_size
                    y_start = row * block_size
                    y_end = (row + 1) * block_size

                    # Assign cluster ID to nodes within this block
                    fitness_scores = []
                    for node in self.Sensors:
                        if (
                            node.layer_number == layer
                            and x_start <= node.xd < x_end
                            and y_start <= node.yd < y_end
                        ):
                            node.cluster_id = cluster_id

                    cluster_nodes = [
                        node for node in self.Sensors if node.cluster_id == cluster_id
                    ]
                    if len(cluster_nodes) == 0:
                        continue
                    fitness_scores = [
                        self.calculate_fitness(node, cluster_id, cluster_nodes)
                        for node in self.Sensors
                        if node.cluster_id == cluster_id
                    ]
                    cluster_head_id = cluster_nodes[
                        fitness_scores.index(min(fitness_scores))
                    ].id
                    for node in self.Sensors:
                        if node.id == cluster_head_id:
                            node.type = "C"
                            self.list_CH.append(node)
                    cluster_id += 1  # Increment cluster ID for the next block

    def select_cluster_heads(self, layer_nodes, layer):

        cl_id = 0
        for cluster in set(node.cluster_id for node in layer_nodes):
            cluster_nodes = [node for node in layer_nodes if node.cluster_id == cluster]

            new_cluster_id = (layer - 1) * self.my_model.clusters_per_layer + cl_id
            fitness_scores = [
                self.calculate_fitness(node, new_cluster_id, cluster_nodes)
                for node in cluster_nodes
            ]
            cluster_head = cluster_nodes[fitness_scores.index(min(fitness_scores))]
            for cluster_node in cluster_nodes:
                cluster_node.cluster_id = new_cluster_id

            cluster_head.type = "C"
            cluster_head.cluster_id = new_cluster_id
            self.list_CH.append(cluster_head)
            cl_id += 1

    def calculate_fitness(self, node, ch_id, cluster_nodes):

        # print("Node:clusterId:",node.cluster_id)

        fitness = 0.00
        distance_to_cluster_center = []
        for cluster_node in cluster_nodes:
            if cluster_node.id == node.id:
                continue
            if cluster_node.id == node.id:
                continue
            distance_to_cluster_center.append(
                (node.xd - cluster_node.xd) ** 2
                + (node.yd - cluster_node.yd) ** 2
                + (node.zd - cluster_node.zd) ** 2
            )

        for cluster_node in cluster_nodes:
            if cluster_node.id == node.id:
                continue
            if cluster_node.id == node.id:
                continue
            distance_to_node = (
                (node.xd - cluster_node.xd) ** 2
                + (node.yd - cluster_node.yd) ** 2
                + (node.zd - cluster_node.zd) ** 2
            )
            fitness += distance_to_node  # Final Fitness Func FF
            # fitness += (
            #     self.my_model.ff_alpha
            #     * (1.0 - distance_to_node / max(distance_to_cluster_center))
            #     + (1.0 - self.my_model.ff_alpha) * node.E / self.my_model.Eo
            # )  # Final Fitness Func FF

        return fitness

    def calculate_fitness_eecrap(self, node, ch_id, cluster_nodes):

        # print("Node:clusterId:",node.cluster_id)

        fitness = 0.00

        for cluster_node in cluster_nodes:
            if cluster_node.id == node.id:
                continue
            distance_to_node = (
                (node.xd - cluster_node.xd) ** 2
                + (node.yd - cluster_node.yd) ** 2
                + (node.zd - cluster_node.zd) ** 2
            )
            if distance_to_node != 0:
                fitness -= self.my_model.ff_alpha / distance_to_node + (
                    1.0 - self.my_model.ff_alpha
                ) * (node.E)
            else:
                fitness += 0

        return fitness

    def perform_cluster_id_new_ch(self, cluster_id):
        cluster_nodes = [node for node in self.Sensors if node.cluster_id == cluster_id]
        fitness_scores = [
            self.calculate_fitness(node, cluster_id, cluster_nodes)
            for node in cluster_nodes
        ]
        cluster_head = cluster_nodes[fitness_scores.index(min(fitness_scores))]
        # print("ok")
        cluster_head.type = "C"
        self.list_CH.append(cluster_head)

    def perform_cluster_id_new_ch_2(self, cluster_id):

        cluster_nodes = [node for node in self.Sensors if node.cluster_id == cluster_id]
        fitness_scores = [
            self.calculate_fitness(node, cluster_id, cluster_nodes)
            for node in self.Sensors
            if node.cluster_id == cluster_id
        ]
        cluster_head_id = cluster_nodes[fitness_scores.index(min(fitness_scores))].id
        for node in self.Sensors:
            if node.id == cluster_head_id:
                node.type = "C"
                self.list_CH.append(node)

    def find_valid_receiver(self, sender):  # for distance based routing
        receivers = [
            node
            for node in self.Sensors
            if node.hop_count < sender.hop_count
            and node.type == "C"
            and node.id != sender.id
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [
            (
                (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                if (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                <= self.my_model.tx_range
                else float("inf")
            )
            for receiver in receivers
        ]
        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        receivers = [
            node
            for node in self.Sensors
            if node.hop_count == sender.hop_count
            and node.type == "C"
            and node.id != sender.id
            and node.E > 0
        ]

        # Calculate distances between the sender and each receiver
        distances = [
            (
                (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                if (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                <= self.my_model.tx_range
                else float("inf")
            )
            for receiver in receivers
        ]

        # Find the index of the nearest receiver
        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

    def find_valid_receiver_2(self, sender):  # for mot func
        # receivers = [
        #     node
        #     for node in self.Sensors
        #     if node.zd >= sender.zd
        #     and node.type == "C"
        #     and node.id != sender.id
        #     and node.layer_number = sender.layer_number
        #     and node.E > 0
        # ]

        receivers = [
            node
            for node in self.Sensors
            if node.zd >= sender.zd
            and node.type == "C"
            and node.id != sender.id
            and node.layer_number == sender.layer_number + 1
            and node.E > 0
        ]

        # Implement Selction Function
        optimal_receiver = None
        min_value = float("inf")  # Initialize with infinity
        alpha = 1
        beta = 0.0
        gamma = 0.0
        base_distance = self.my_model.z * sqrt(3) / 5
        for receiver in receivers:
            depth_diff = abs(sender.zd - receiver.zd) / abs(self.my_model.z)
            e_res = receiver.E / self.my_model.Eo
            distance = (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            ) ** 0.5
            distance = distance / base_distance

            # Calculate the value of the expression
            value = alpha * e_res + beta * distance + gamma * depth_diff

            # Update optimal receiver if the calculated value is lower
            if value < min_value:
                min_value = value
                optimal_receiver = receiver

        if optimal_receiver != None:
            return optimal_receiver

        receivers = [
            node
            for node in self.Sensors
            if node.hop_count < sender.hop_count
            and node.type == "C"
            and node.id != sender.id
            and node.zd > sender.zd
            and node.E > 0
        ]

        min_value = float("inf")  # Initialize with infinity

        base_distance = self.my_model.z * sqrt(3)
        for receiver in receivers:
            depth_diff = abs(sender.zd - receiver.zd) / abs(self.my_model.z)
            e_res = receiver.E / self.my_model.Eo
            distance = (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            ) ** 0.5
            distance = distance / base_distance

            # Calculate the value of the expression
            value = alpha * e_res + beta * distance + gamma * depth_diff

            # Update optimal receiver if the calculated value is lower
            if value < min_value:
                min_value = value
                optimal_receiver = receiver

        if optimal_receiver != None:
            return optimal_receiver

        receivers = [
            node
            for node in self.Sensors
            if node.hop_count == sender.hop_count
            and node.type == "C"
            and node.id != sender.id
            and node.zd > sender.zd
            and node.E > 0
        ]

        min_value = float("inf")  # Initialize with infinity

        base_distance = self.my_model.z * sqrt(3)
        for receiver in receivers:
            depth_diff = abs(sender.zd - receiver.zd) / abs(self.my_model.z)
            e_res = receiver.E / self.my_model.Eo
            distance = (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            ) ** 0.5
            distance = distance / base_distance

            # Calculate the value of the expression
            value = alpha * e_res + beta * distance + gamma * depth_diff

            # Update optimal receiver if the calculated value is lower
            if value < min_value:
                min_value = value
                optimal_receiver = receiver

        return optimal_receiver

    def find_valid_receiver_3(self, sender):  # for dvor
        receivers = [
            node
            for node in self.Sensors
            if node.type == "C"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [
            (
                (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                if (
                    (
                        (sender.xd - receiver.xd) ** 2
                        + (sender.yd - receiver.yd) ** 2
                        + (sender.zd - receiver.zd) ** 2
                    )
                    ** 0.5
                )
                <= self.my_model.tx_range
                else float("inf")
            )
            for receiver in receivers
        ]

        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

    def find_valid_receiver_5(self, sender):  # for dvor+hop count
        receivers = [
            node
            for node in self.Sensors
            if node.type == "C"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.layer_number > sender.layer_number
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [
            (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            )
            ** 0.5
            for receiver in receivers
        ]

        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

    def find_valid_receiver_6(self, sender):  # for dvor+hop count
        receivers = [
            node
            for node in self.Sensors
            if node.type == "C"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [receiver.E for receiver in receivers]

        if distances:
            nearest_receiver_index = distances.index(max(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

    def find_valid_receiver_7(self, sender): #dvor
        receivers = [
            node
            for node in self.Sensors
            if node.type == "N"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [
            (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            )
            ** 0.5
            for receiver in receivers
        ]

        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

    def find_valid_receiver_4(self, sender):  # for dbr
        receivers = [
            node
            for node in self.Sensors
            if node.type == "N"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.E > 0
        ]

        # Implement Selction Function
        # distances = [receiver.E for receiver in receivers]
        distances = [
            abs(receiver.zd - sender.zd)
            for receiver in receivers
            if (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            )
            ** 0.5
            <= self.my_model.tx_range
        ]

        if distances:
            nearest_receiver_index = distances.index(min(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

        receivers = [
            node
            for node in self.Sensors
            if node.type == "C"
            and node.id != sender.id
            and node.zd >= sender.zd
            and node.E > 0
        ]

        # Implement Selction Function
        distances = [receiver.E for receiver in receivers]

        if distances:
            nearest_receiver_index = distances.index(max(distances))
            nearest_receiver = receivers[nearest_receiver_index]
            return nearest_receiver

        return None

    def find_nearest_sink(self, sender):
        receivers = [node for node in self.Sensors if node.type == "S" and node.E >= 0]

        distances = [
            (
                (sender.xd - receiver.xd) ** 2
                + (sender.yd - receiver.yd) ** 2
                + (sender.zd - receiver.zd) ** 2
            )
            ** 0.5
            for receiver in receivers
        ]
        if len(distances) == 0:
            return self.Sensors[self.my_model.n]
        nearest_receiver_index = distances.index(min(distances))
        nearest_receiver = receivers[nearest_receiver_index]
        return nearest_receiver
