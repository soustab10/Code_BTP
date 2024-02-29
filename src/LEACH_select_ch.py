from src.LEACH_create_basics import *


def zeros(row, column):
    re_list = []
    for x in range(row):
        temp_list = [0 for _ in range(column)]
        if row == 1:
            re_list.extend(temp_list)
        else:
            re_list.append(temp_list)

    return re_list


def start(sensors: list[Sensor], my_model, round_number: int):
    CH = []
    # countCHs = 0 # no use
    n = my_model.n

    # numRx = myModel.numRx
    # dr = myModel.dr
    # CH_selected_arr = zeros(numRx, numRx)

    # sink can't be a CH
    for sensor in sensors[:-1]:

        # # % these are the circle (x,y) for this node
        # row_circle_of_node = -1
        # col_circle_of_node = -1
        # br = 0
        #
        # # % checking in which circle this node lies
        # for row in range(numRx):
        #     for column in range(numRx):
        #         if (sqrt((Sensors[i].xd - circlex[row][column]) ^ 2 +
        #                  (Sensors[i].yd - circley[row][column]) ^ 2) <= dr / 2):
        #             row_circle_of_node = row
        #             col_circle_of_node = column
        #
        #             br = 1
        #             break
        #
        #     if br == 1:
        #         break
        #
        # # % if this node is not in any circle then also skip
        # if br == 0:
        #     continue
        #
        # # % if CH of this circle has already been chosen, then skip
        # if CH_selected_arr[row_circle_of_node][col_circle_of_node] == 1:
        #     continue

        # If current sensor has energy left and has not been CH before And it is not dead
        # todo: keep either 'sensor.E > 0' or 'sensor.df == 0'

        if sensor.E > 0 and sensor.G <= 0:
            # Election of Cluster Heads
            temp_rand = random.uniform(0, 1)
            value = my_model.p / (1 - my_model.p * (round_number % round(1 / my_model.p)))
            print(f'for {sensor.id}, temprand = {temp_rand}, value = {value}')
            if temp_rand <= value:
                print(f"Adding {sensor.id} to CH")
                CH.append(sensor.id)
                sensor.type = 'C'
                sensor.G = round(1 / my_model.p) - 1

                # # mark this cirle now that it has a CH
                # CH_selected_arr(row_circle_of_node, col_circle_of_node) = 1

    return CH


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
