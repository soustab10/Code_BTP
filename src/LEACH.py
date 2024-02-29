import pprint
from math import *
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

from src import LEACH_create_basics
from src import LEACH_select_ch
from src import findReceiver
from src import find_sender
from src import join_to_nearest_ch
from src import reset_sensors
from src import send_receive_packets
from src import LEACH_plotter


def var_pp(stuff):
    pass
    # Todo: UNCOMMENT
    # prettty_prrint = pprint.PrettyPrinter(indent=1)
    # for x in stuff:
    #     prettty_prrint.pprint(vars(x))


def pp(stuff):
    print(stuff)
    # Todo: UNCOMMENT
    # prettty_prrint = pprint.PrettyPrinter(indent=4)
    # prettty_prrint.pprint(stuff)


# #################################################


def zeros(row, column):
    re_list = []
    for x in range(row):
        temp_list = [0 for _ in range(column)]
        if row == 1:
            re_list.extend(temp_list)
        else:
            re_list.append(temp_list)

    return re_list


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
        
        # self.__print_sensors()
        # self.__start_simulation()
        self.__main_loop()
        # self.__plot_sensors()s
        # Todo: all plotting should be done in Leach_plotter file
        # plt.xlim(left=0, right=self.my_model.rmax)
        # plt.ylim(bottom=0, top=self.n)
        # plt.plot(self.alive_sensors)
        # plt.title("Life time of sensor nodes")
        # plt.xlabel('Rounds')
        # plt.ylabel('No. of live nodes')
        # # plt.ioff()
        # plt.show()

        # plt.xlim(left=0, right=self.my_model.rmax)
        # plt.ylim(bottom=0, top=self.n * self.my_model.Eo)
        # plt.plot(self.sum_energy_left_all_nodes)
        # plt.title("Total residual energy ")
        # plt.xlabel('Rounds')
        # plt.ylabel('Energy (J)')
        # plt.show()

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

            print(f"first dead in round: {round_number}")

    def __create_sen(self):
        print("##########################################")
        print("############# Create Sensors #############")
        print("##########################################")
        print()

        # Create a random scenario & Load sensor Location
        # configure sensors
        self.Sensors = LEACH_create_basics.create_sensors(self.my_model)

        for sensor in self.Sensors[:-1]:
            self.initEnergy += sensor.E

        # We will have full energy in start
        self.sum_energy_left_all_nodes[0] = self.initEnergy
        self.avg_energy_All_sensor[0] = self.initEnergy / self.n

        # todo: test
        var_pp(self.Sensors)
        print("self.initEnergy", self.initEnergy)
        print("----------------------------------------------")

    def __plot_sensors(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        x_vals = [node.xd for node in self.Sensors if node.type == 'N']
        y_vals = [node.yd for node in self.Sensors if node.type == 'N']
        z_vals = [node.zd for node in self.Sensors if node.type == 'N']
        x_vals_ch = [node.xd for node in self.Sensors if node.type == 'C']
        y_vals_ch = [node.yd for node in self.Sensors if node.type == 'C']
        z_vals_ch = [node.zd for node in self.Sensors if node.type == 'C']

        sink_x_vals = [x_vals[-1]]
        sink_y_vals = [y_vals[-1]]
        sink_z_vals = [z_vals[-1]]

        x_vals = x_vals[:-1]
        y_vals = y_vals[:-1]
        z_vals = z_vals[:-1]

        ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o", label="Nodes")
        ax.scatter(x_vals_ch, y_vals_ch, z_vals_ch, c="red", marker="x", label="CHs")

        # Plot blue planes at specified heights
        for height in self.layer_heights:
            x_plane = np.linspace(min(x_vals), max(x_vals), 100)
            y_plane = np.linspace(min(y_vals), max(y_vals), 100)
            x_plane, y_plane = np.meshgrid(x_plane, y_plane)
            z_plane = np.full_like(x_plane, height)
            ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.1, color="blue")

        ax.scatter(
            sink_x_vals,
            sink_y_vals,
            sink_z_vals,
            c="red",
            marker="^",
            label="Sink Nodes",
        )

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("3D UWSN")

        plt.legend()
        plt.show()

    def __print_sensors(self):
        print("##########################################")
        print("############# Print Sensors #############")
        print("##########################################")
        print()
        ct=0
        num_clusters = set()
        for sensor in self.Sensors:
            print(sensor.id, sensor.xd, sensor.yd, sensor.zd, sensor.layer_number, sensor.cluster_id, sensor.type)
            num_clusters.add(sensor.cluster_id)
            if(sensor.type == 'C'):
                ct+=1
            
        print("Total Number of CHs:", ct)
        print("Total Number of Clusters:", len(num_clusters))
        print("----------------------------------------------")
        
    def plot_clusters(self, layer_number=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cluster_sensors = {}
        ch_coordinates = {'x': [], 'y': [], 'z': []}

        for sensor in self.Sensors:
            if sensor.cluster_id not in cluster_sensors:
                cluster_sensors[sensor.cluster_id] = {'x': [], 'y': [], 'z': []}

            cluster_sensors[sensor.cluster_id]['x'].append(sensor.xd)
            cluster_sensors[sensor.cluster_id]['y'].append(sensor.yd)
            cluster_sensors[sensor.cluster_id]['z'].append(sensor.zd)

            if sensor.type == 'C':
                ch_coordinates['x'].append(sensor.xd)
                ch_coordinates['y'].append(sensor.yd)
                ch_coordinates['z'].append(sensor.zd)

        for cluster_id, coordinates in cluster_sensors.items():
            ax.scatter(coordinates['x'], coordinates['y'], coordinates['z'], label=f'Cluster {cluster_id}')

        ax.scatter(ch_coordinates['x'], ch_coordinates['y'], ch_coordinates['z'], c='black', marker='x', s=100, label='Cluster Heads')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Sensor Network by Cluster with Cluster Heads')
        plt.legend()
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

        self.sender = [
            self.n
        ]  # List of senders, for start_sim, sink will send hello packet to all
        self.receivers = [
            _ for _ in range(self.n)
        ]  # List of senders, for start_sim, All nodes will receive from sink

        # todo: test
        print("Senders: ", end="")
        pp(self.sender)
        print("Receivers: ", end="")
        pp(self.receivers)
        print()

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
        print("self.srp", self.srp)
        print("self.rrp", self.rrp)
        print("self.sdp", self.sdp)
        print("self.rdp", self.rdp)
        print("Sensors: \n", var_pp(self.Sensors))

        # Save metrics, Round 0 is initialization phase where all nodes send routing packets (hello) to Sink as above
        self.SRP[0] = self.srp
        self.RRP[0] = self.rrp
        self.SDP[0] = self.sdp
        self.RDP[0] = self.rdp

        # todo: test
        print("self.SRP", self.SRP)
        print("self.RRP", self.RRP)
        print("self.SDP", self.SDP)
        print("self.RDP", self.RDP)

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
            print(
                "Sensors: ",
            )
            var_pp(self.Sensors)

            # ############# cluster head election #############
            
            self.__cluster_head_selection_phase(round_number)
            self.no_of_ch = len(self.list_CH)  # Number of CH in per period

            # self.__steady_state_phase()

            # if sensor is dead
            # self.__check_dead_num(round_number)

            # self.__statistics(round_number)

            # if all nodes are dead or only sink is left, exit
            if len(self.dead_num) >= self.n:
                self.lastPeriod = round_number
                print(f"all dead (dead={len(self.dead_num)}) in round {round_number}")
                break

    def __cluster_head_selection_phase(self, round_number):
        print("#################################################")
        print("############# cluster head election #############")
        print("#################################################")
        print()
        print("Clusters of Current Round:", self.list_CH)
        # Selection Candidate Cluster Head Based on LEACH Set-up Phase
        # self.list_CH stores the id of all CH in current round

        self.list_CH = self.perform_clustering(self.my_model.clusters_per_layer)
        # if(round_number == 1):
        #     self.list_CH = LEACH_select_ch.start(self.Sensors, self.my_model, round_number)
        # else:
        # # Check if energy of each existing cluster head is below threshold
        #     clusters_to_re_elect = []
        #     for cluster_head_id in self.list_CH:
        #         cluster_head = next((node for node in self.Sensors if node.node_id == cluster_head_id), None)
        #         if cluster_head and cluster_head.energy < 0.5:  # Adjust ENERGY_THRESHOLD as needed
        #             clusters_to_re_elect.append(cluster_head.cluster_id)

        #     # Re-elect cluster heads only for clusters with energy below threshold
        #     if clusters_to_re_elect:
        #         self.list_CH = LEACH_select_ch.start(self.Sensors, self.my_model, round_number, clusters_to_re_elect)
        #         self.no_of_ch = len(self.list_CH)
        # #add checker if CH has energy below threshold
        # #store cluster ids of all such clusters
        # #only re elect chs of those clusters
        # self.list_CH = LEACH_select_ch.start(self.Sensors, self.my_model, round_number)
        self.no_of_ch = len(self.list_CH)

        # todo: test
        print("Cluster Heads: ", self.list_CH)
        
        # if round_number == 1 and len(self.list_CH) == 0:
        #     exit("EXIT, no CH in initial round")
        print()

        # #########################################################################################
        # ############# Broadcasting CHs to All Sensors that are in Radio Range of CH #############
        # #########################################################################################
        self.__broadcast_cluster_head()

        # ######################################################
        # ############# Sensors join to nearest CH #############
        # ######################################################
        # updates dist2ch & cluster_id in node
        # join_to_nearest_ch.start(self.Sensors, self.my_model, self.list_CH)

        # todo: test
        self.__print_sensors()
        self.plot_clusters()
        # ########################################
        # ############# plot Sensors #############
        # ########################################
        # Todo: plot here

        # ##############################################################
        # ############# end of cluster head election phase #############
        # ##############################################################

        print("##############################################################")
        print("############# end of cluster head election phase #############")
        print("##############################################################")

    def __broadcast_cluster_head(self):
        print(
            "#########################################################################################"
        )
        print(
            "############# Broadcasting CHs to All Sensors that are in Radio Range of CH #############"
        )
        print(
            "#########################################################################################"
        )
        print()

        # Broadcasting CH x to All Sensors that are in Radio Rage of x. (dont broadcast to sink)
        # Doing this for all CH
        for cluster_head in self.list_CH:
            # todo: test
            print(f"for cluster head: {cluster_head}")
            self.receivers: list = findReceiver.start(
                self.Sensors,
                self.my_model,
                sender=cluster_head,
                sender_rr=self.Sensors[cluster_head.id].RR,
            )

            # todo: test
            print("\nsender (or CH): ", cluster_head)
            print("self.Receivers: ", end="")
            print(self.receivers)

            # we require the sender parameter of sendReceivePackets.start to be a list.
            self.srp, self.rrp, self.sdp, self.rdp = send_receive_packets.start(
                self.Sensors,
                self.my_model,
                [cluster_head],
                self.receivers,
                self.srp,
                self.rrp,
                self.sdp,
                self.rdp,
                packet_type="Hello",
            )

            # todo: test
            print("self.srp", self.srp)
            print("self.rrp", self.rrp)
            print("self.sdp", self.sdp)
            print("self.rdp", self.rdp)
            print(
                "Sensors: ",
            )
            var_pp(self.Sensors)
            print()

    def __steady_state_phase(self):
        print("##############################################")
        print("############# steady state phase #############")
        print("##############################################")
        print()

        for i in range(
            self.my_model.NumPacket
        ):  # Number of Packets to be sent in steady-state phase

            # ########################################
            # ############# plot Sensors #############
            # ########################################
            # todo: Plot here

            # #############################################################
            # ############# All sensor send data packet to CH #############
            # #############################################################
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
                var_pp(self.Sensors)
                print()

        # ####################################################################################################
        # ############# send Data packet directly from nodes(that aren't in any cluster) to Sink #############
        # ####################################################################################################
        print(
            "####################################################################################################"
        )
        print(
            "############# send Data packet directly from nodes(that aren't in any cluster) to Sink #############"
        )
        print(
            "####################################################################################################"
        )

        for sender in self.Sensors:
            # if the node has sink as its CH but it's not sink itself and the node is not dead
            if sender.cluster_id == self.n and sender.id != self.n and sender.E > 0:
                self.receivers = [self.n]  # Sink
                sender = [sender.id]

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
        print(
            "###################################################################################"
        )
        print(
            "############# Send Data packet from CH to Sink after Data aggregation #############"
        )
        print(
            "###################################################################################"
        )
        print()

        # todo: test
        print("senders (or CH) = ", self.list_CH)

        for sender in self.list_CH:
            self.receivers = [self.n]  # Sink

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
            var_pp(self.Sensors)
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
        for sensor in self.Sensors[:-1]:
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

        # todo: maybe this is related to graph?
        # title(sprintf('Round=##d,Dead nodes=##d', round_number, deadNum))

        # todo: test
        print("round number:", round_number)
        print("len(self.SRP)", len(self.SRP))
        print("self.SRP", self.SRP)
        print("self.RRP", self.RRP)
        print("self.SDP", self.SDP)
        print("self.RDP", self.RDP)
        print("----------------------------------------------")

        # print('self.total_energy_dissipated', self.total_energy_dissipated)
        # print('self.AllSensorEnergy', self.AllSensorEnergy)
        print("self.sum_dead_nodes", self.sum_dead_nodes)
        print("self.ch_per_round", self.ch_per_round)
        print("self.alive_sensors", self.alive_sensors)
        print("self.sum_energy_all_nodes", self.sum_energy_left_all_nodes)
        print("self.avg_energy_All_sensor", self.avg_energy_All_sensor)
        print("self.consumed_energy", self.consumed_energy)
        print("self.Enheraf", self.Enheraf)

        print(
            "Sensors: ",
        )
        var_pp(self.Sensors)
        print("----------------------------------------------")

    def perform_clustering(self, k):
        total_depth = abs(self.my_model.z_range[0] - self.my_model.z_range[1])
        CH = []
        
        # Iterate through layers
        for layer in set(node.layer_number for node in self.Sensors):
            layer_nodes = [node for node in self.Sensors if node.layer_number == layer]
            # print(layer_nodes)
            if not layer_nodes:
                continue  # Skip layers with no nodes

            # Combine node coordinates into a single array for clustering
            node_coordinates = [[node.xd, node.yd, node.zd] for node in layer_nodes]
            print(node_coordinates)
            # Use KMeans clustering with explicit n_init
            kmeans = KMeans(
                n_clusters=k, n_init=10, random_state=0
            )  # Set n_init explicitly
            kmeans.fit(node_coordinates)
            print(kmeans.labels_)
            # Assign each node to its respective cluster
            
            for i, node in enumerate(layer_nodes):
                node.cluster_id = kmeans.labels_[i]

            # Store the cluster centers
            self.cluster_centers = kmeans.cluster_centers_

            # Select cluster heads based on a fitness function
            
            self.select_cluster_heads(layer_nodes, layer, CH)

        
        
        return CH

    def select_cluster_heads(self, layer_nodes, layer, CH):
        
        cl_id = 0
        for cluster in set(node.cluster_id for node in layer_nodes):
            cluster_nodes = [node for node in layer_nodes if node.cluster_id == cluster]
            fitness_scores = [self.calculate_fitness(node) for node in cluster_nodes]
            cluster_head = cluster_nodes[fitness_scores.index(min(fitness_scores))]
            for cluster_node in cluster_nodes:
                cluster_node.cluster_id = layer*self.my_model.clusters_per_layer+cl_id

            cluster_head.type = "C"
            cluster_head.cluster_id = layer*self.my_model.clusters_per_layer+cl_id
            CH.append(cluster_head)
            cl_id+=1
            

    def calculate_fitness(self, node):
        # Assuming k-means clustering has been performed
        # Calculate the distance from the node to its cluster center
        cluster_center = self.cluster_centers[node.cluster_id]
        distance_to_cluster_center = math.sqrt(
            (node.xd - cluster_center[0]) ** 2
            + (node.yd - cluster_center[1]) ** 2
            + (node.zd - cluster_center[2]) ** 2
        )

        fitness = (
            self.my_model.ff_alpha * distance_to_cluster_center
            + (1 - self.my_model.ff_alpha) * node.E
        )  # Final Fitness Func FF
        return fitness
