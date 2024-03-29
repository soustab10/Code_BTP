import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import *
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class Node:
    def __init__(self, node_id, x, y, z, initial_energy):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.depth = abs(z)
        self.layer_number = self.get_layer_number()
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.cluster_id = 0 
        self.is_CH = False 
        
    def get_layer_number(self):
        layer_heights = [0, -80, -170, -270, -380, -500]
        for i, height in enumerate(layer_heights[::-1], start=0):
            if self.z <= height:
                return i
        return 0
    
    def calculate_distance(self, other_node):
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2 + (self.z - other_node.z)**2)

    
    def calculate_ch_cost(self, alpha, total_depth):
        d_div = abs(self.z / total_depth)
        cost_ch = alpha * (self.current_energy / self.initial_energy) + (1 - alpha) * (1 - d_div)
        return cost_ch

    def __str__(self):
        return f"Node {self.node_id}: ({self.x}, {self.y}, {self.z}, at layer {self.layer_number})"
    
    
class ClusterHead:
    def __init__(self, node_id, x, y, z, layer_number,current_energy,cluster_id):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.z = z
        self.layer_number = layer_number
        self.current_energy = current_energy
        self.hop_count = num_layers - layer_number +1
        self.cluster_id = cluster_id
        
class Sink:
    def __init__(self, sink_id, x, y, z):
        self.sink_id = sink_id
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"Sink {self.sink_id}: ({self.x}, {self.y}, {self.z})"
    
    
class UnderwaterWSN:
    def __init__(self, num_nodes,num_sinks, x_range, y_range, z_range, tx_range, num_clusters,initial_energy,ff_alpha):
        self.nodes = []
        self.sinks = []
        self.num_nodes = num_nodes
        self.num_sinks = num_sinks
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.ff_alpha = ff_alpha
        self.tx_range = tx_range
        self.num_clusters = num_clusters
        self.initial_energy = initial_energy
        self.clusters = []
        self.cluster_centers = []
        self.cluster_heads = []
        self.rounds=rounds_max

    def deploy_nodes_randomly(self):
        for i in range(self.num_nodes):
            x = random.uniform(self.x_range[0], self.x_range[1])
            y = random.uniform(self.y_range[0], self.y_range[1])
            z = random.uniform(self.z_range[0], self.z_range[1])
            node = Node(i + 1, x, y, z, self.initial_energy)
            self.nodes.append(node)

    def deploy_sinks(self, num_sinks):
        if num_sinks > 0:
            # Specific x and y coordinates for sinks
            x_coords = [50, 150, 250, 350, 450]
            y_coords = [50, 150, 250, 350, 450]
            z = self.z_range[1]

            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    sink = Sink(i * len(y_coords) + j + 1, x, y, z)
                    self.sinks.append(sink)
                    
    

    def select_cluster_numbers(self,k_range=(2,100)):
       
    
        # Extract features from nodes
        data = np.array([[node.x, node.y, node.z, node.current_energy] for node in self.nodes if node.layer_number == 2])

        model = KMeans(n_init=100, random_state=42)

        # k is the range of the number of clusters.
        visualizer = KElbowVisualizer(model, k=k_range, timings=True)

        # Fit data to visualizer
        visualizer.fit(data)

        # Finalize and render figure
        visualizer.show()

                       
    def display_nodes(self):
        for node in self.nodes:
            print(node)

    def display_sinks(self):
        for sink in self.sinks:
            print(sink)
            
    def distance_between_nodes(self, node1, node2):
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

    def calculate_cost(self, alpha, initial_energy, current_energy, z, total_depth):
        d_div = abs(z / total_depth)
        cost_ch = alpha * (current_energy / initial_energy) + (1 - alpha) * (1 - d_div)
        return cost_ch

    def plot_nodes(self, layer_heights):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_vals = [node.x for node in self.nodes]
        y_vals = [node.y for node in self.nodes]
        z_vals = [node.z for node in self.nodes]

        ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o', label='Nodes')

        # Plot pink planes at specified heights
        for height in layer_heights:
            x_plane = np.linspace(min(x_vals), max(x_vals), 100)
            y_plane = np.linspace(min(y_vals), max(y_vals), 100)
            x_plane, y_plane = np.meshgrid(x_plane, y_plane)
            z_plane = np.full_like(x_plane, height)
            ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.3, color='pink')

        sink_x_vals = [sink.x for sink in self.sinks]
        sink_y_vals = [sink.y for sink in self.sinks]
        sink_z_vals = [sink.z for sink in self.sinks]
        ax.scatter(sink_x_vals, sink_y_vals, sink_z_vals, c='pink', marker='^', label='Sink Nodes')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D UWSN')

        plt.legend()
        plt.show()
    
    def calculate_layer(self, z):
        # Helper function to calculate the layer number based on the z-coordinate
        layer_heights = [0, -80, -170, -270, -380, -500]
        for i, height in enumerate(layer_heights, start=1):
            if z > height:
                return i
        return len(layer_heights)  # If z is below the lowest layer

    def perform_clustering(self, k):
        total_depth = abs(self.z_range[0] - self.z_range[1])

        # Iterate through layers
        for layer in set(node.layer_number for node in self.nodes):
            layer_nodes = [node for node in self.nodes if node.layer_number == layer]

            if not layer_nodes:
                continue  # Skip layers with no nodes

            # Combine node coordinates into a single array for clustering
            node_coordinates = [[node.x, node.y, node.z] for node in layer_nodes]

            # Use KMeans clustering with explicit n_init
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # Set n_init explicitly
            kmeans.fit(node_coordinates)

            # Assign each node to its respective cluster
            for i, node in enumerate(layer_nodes):
                node.cluster_id = kmeans.labels_[i]

            # Store the cluster centers
            self.cluster_centers.extend(kmeans.cluster_centers_)

            # Select cluster heads based on a fitness function
            cluster_id_head = 1
            self.select_cluster_heads(layer_nodes,cluster_id_head)

    def select_cluster_heads(self, layer_nodes,cluster_id_head):
        
        for cluster in set(node.cluster_id for node in layer_nodes):
            cluster_nodes = [node for node in layer_nodes if node.cluster_id == cluster]
            fitness_scores = [self.calculate_fitness(node) for node in cluster_nodes]
            cluster_head = cluster_nodes[fitness_scores.index(min(fitness_scores))]
            self.clusters.append(cluster_nodes)
            cluster_head.is_CH = True
            cluster_head.cluster_id = cluster_id_head            
            self.cluster_heads.append(ClusterHead(cluster_head.node_id, cluster_head.x, cluster_head.y, cluster_head.z, cluster_head.layer_number,cluster_head.current_energy,cluster_id_head))
            cluster_id_head+=1

    def calculate_fitness(self, node):
        # Assuming k-means clustering has been performed
        # Calculate the distance from the node to its cluster center
        cluster_center = self.cluster_centers[node.cluster_id]
        distance_to_cluster_center = math.sqrt((node.x - cluster_center[0])**2 +
                                              (node.y - cluster_center[1])**2 +
                                              (node.z - cluster_center[2])**2)

        d_div = abs(node.z / total_depth)
        fitness = ff_alpha*distance_to_cluster_center + (1 - ff_alpha)*node.current_energy  #Final Fitness Func FF
        return fitness
       
    def perform_clustering_with_fitness(self, cluster_per_layer, alpha):
        total_depth = abs(self.z_range[0] - self.z_range[1])

        # Function to calculate fitness for a node
        def calculate_fitness(node, total_depth,layer_nodes):
            fitness = 0
            for other_node in layer_nodes:
                dist = node.calculate_distance(other_node)
                if(dist!=0):
                    fitness += (alpha * 1/node.calculate_distance(other_node) ) + (1-alpha)*node.current_energy
           
            return fitness

        for layer in range(1, 6):  # Considering one layer at a time (assuming there are 5 layers)
            layer_nodes = [node for node in self.nodes if node.layer_number == layer]

            if not layer_nodes:
                continue  # Skip layers with no nodes

            clusters_in_layer = []
            # Form clusters with cluster_per_layer nodes in each layer
            for _ in range(4):  # Ensure each layer has four clusters
                if not layer_nodes:
                    break  # Break if there are no more nodes in the layer
                optimal_cluster_head = max(layer_nodes, key=lambda node: calculate_fitness(node, total_depth,layer_nodes))
                cluster_nodes = [node for node in layer_nodes if node != optimal_cluster_head][:cluster_per_layer - 1]
                cluster_nodes.append(optimal_cluster_head)
                clusters_in_layer.append(cluster_nodes)
                layer_nodes = [node for node in layer_nodes if node not in cluster_nodes]

            self.clusters.extend(clusters_in_layer)
            
            
    

    def print_clusters(self):
        for i, (cluster_nodes, cluster_head) in enumerate(zip(self.clusters, self.cluster_heads), start=1):
            print(f"Cluster {i} (Layer {cluster_nodes[0].layer_number}):")
            print(f"  CH: {cluster_head.node_id}")
            max_dist=0
            for node in cluster_nodes:
                print(f"    Node {node.node_id} - Layer {node.layer_number}")
                max_dist=max(max_dist,self.distance_between_nodes(node,cluster_head))
                
            print(f"    Max Distance to CH: {max_dist}")

    def plot_clusters(self, layer_number=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, cluster_nodes in enumerate(self.clusters, start=1):
            if layer_number is not None and any(node.layer_number != layer_number for node in cluster_nodes):
                continue  # Skip clusters not belonging to the specified layer

            x_vals = [node.x for node in cluster_nodes]
            y_vals = [node.y for node in cluster_nodes]
            z_vals = [node.z for node in cluster_nodes]
            ax.scatter(x_vals, y_vals, z_vals, marker='o', label=f'Cluster {i}')


        
        for ch in self.cluster_heads:
            ax.scatter(ch.x, ch.y, ch.z, c='black', marker='x', label=f'Cluster Head {ch.node_id} - Layer {ch.layer_number}')
        for sink in self.sinks:
            ax.scatter(sink.x, sink.y, sink.z, c='pink', marker='^', label=f'Sink {sink.sink_id}')
            
        

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'3D UWSN Clusters with Sinks - Layer {layer_number}' if layer_number is not None else '3D UWSN Clusters with Sinks')

        # plt.legend()
        plt.show()
        
        
    # Routing Phase
    def initiate_signal(self, source_node):
        # Pass signal to cluster head
        cluster_head = next((node for node in self.nodes if node.cluster_id == source_node.cluster_id and node.is_CH and node.layer_number == source_node.layer_number), None)
        # print(cluster_head)
        if cluster_head:
            distance_to_cluster_head = self.distance_between_nodes(source_node, cluster_head)
            print(cluster_head,distance_to_cluster_head,"Layer:",source_node.layer_number,cluster_head.layer_number)
            if distance_to_cluster_head < self.tx_range and source_node.current_energy > 0:  # Example transmission range and current_energy threshold
                print(f"Node {source_node.node_id} passing signal to Cluster Head {cluster_head.node_id}")
                source_node.current_energy -= e_tx*packet_size
                

                # Pass signal to nearest sink
                nearest_sink = min(self.sinks, key=lambda sink: self.distance_between_nodes(cluster_head, sink))
                distance_to_sink = self.distance_between_nodes(cluster_head, nearest_sink)
                if cluster_head.current_energy > 0:  # Example transmission range and current_energy threshold
                    print(f"Cluster Head {cluster_head.node_id} sending signal to Sink {nearest_sink.sink_id}")
                    cluster_head.current_energy -= e_tx*packet_size
                    

    def run_simulation(self):
        for node in self.nodes:
            self.initiate_signal(node)
    

    




underwater_wsn = UnderwaterWSN(num_nodes, num_sinks, x_range, y_range, z_range,tx_range,num_clusters,e_inital,ff_alpha)
underwater_wsn.deploy_nodes_randomly()
underwater_wsn.deploy_sinks(num_sinks)
# underwater_wsn.display_nodes()
# underwater_wsn.display_sinks()

# underwater_wsn.select_cluster_numbers()
# underwater_wsn.plot_nodes(layer_heights)

# underwater_wsn.perform_clustering_with_fitness(int(num_clusters), ff_alpha)
underwater_wsn.perform_clustering(int(cluster_per_layer))

underwater_wsn.print_clusters()
underwater_wsn.plot_clusters(layer_number=1)

# underwater_wsn.run_simulation()
