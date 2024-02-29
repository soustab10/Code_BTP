import numpy as np
from math import *

num_nodes = 500
num_sinks=16
x_range = (0, 500) 
y_range = (0, 500)
z_range = (-500, 0)
total_depth = 500

tx_range = 100
num_clusters = 50

e_inital = 5
packet_size = 200
num_bits = 200

f = 10*1000
data_rate=35000

f_khz = 10
alpha_f = 0.11 * f_khz**2 / (1 + f_khz**2) + 44 * f_khz**2 / (4100 + f_khz**2) + 0.003 + 2.75e-4 * f_khz**2
e_elec = 50 * 0.000000001 #energy reqd to run sensor
e_tx = 10 * (10 ** -12) #energy reqd to transfer one bit
e_rx = 10 * (10 ** -12) #energy reqd to receive one bit
# Transmit Amplifier types
efs = 10e-11 #energy loss due to medium
emp = 0.0013 * (10e-12)
# Data Aggregation Energy
eda = 5 * 0.000000001

do = sqrt(efs / emp) * sqrt(num_nodes/(2*3.142))

attn = 10**(alpha_f/10)

speed_sound = 1500

# Maximum number of rounds
rounds_max = 100
curr_round=0

#Total Energies
#define in program
#Tot energy reqd to send n bits data = (num_bits/data_rate) * e_tx * (dist ** 1.5) * (attn** dist)
#Tot energy reqd to receive n bits data = (num_bits/data_rate) * e_rx 
#Electric data consumed  = num_bits*e_elec


#Layer
num_layers=5
layer_heights = [0,-80,-170,-270,-380,-500]

#If all layers are considered to be same height, layer_height = tx_range
# layer_heights = [0,100,200,300,400,500]



# Number of Clusters
L = y_range[1]
M = (3**0.5) * L
# num_clusters = (M*M*num_bits*efs/(10*L*L))**0.60
num_clusters = 18*5
cluster_size = tx_range
cluster_per_layer = (num_clusters/num_layers)

#Fitness function
ff_alpha = 0.5



# # Computation of do/
# do = np.sqrt(Efs / Emp)
# packets_TO_BS = 0
# np.set_printoptions(precision=15)

# alpha = 4
# beta = 2
# gama = 2
# delta = 4
# Elec = 50 * 0.000000001  # Eelec = 50nJ/bit energy transfer and receive
# Efs = 10 * 0.000000000001  # energy free space
# Emp = 0.0013 * 0.000000000001  # energy multi-path
# Kbit = 2000  # size
# CH_Kbit = 200  # Adv. advertisement msg is 25 bytes long i.e. 200 bits
# Eda = 5 * 0.000000001  # data aggregation nj/bit/signal
# MaxInterval = 3  # TIME INTERVAL

# oldnt = np.zeros(n + 1)
# threshold = 35
# Eresx = np.zeros(n + 1)
# Eresy = np.zeros(n + 1)
# summ = np.zeros(n + 1)
# sumation = 0
# permon = np.full(n + 1, 0.0001)
# neighbour = np.zeros(n + 1)
# pr = np.zeros(n + 1)

# SinkID = n + 1
# InitialEnergy = 0.5
# Etx = 0
# Erx = 0
# SUM_Total_Energy = 0
# Threshold = 0
# I = 0
# dpermon = 0
# roo = 0.02
# Rrout = 0
# Rounds = 0
# d0 = np.sqrt(Efs / Emp)
# actual_rounds = 0
# DEAD = np.zeros(rmax + 1)
# DEAD_N = np.zeros(rmax + 1)
# DEAD_A = np.zeros(rmax + 1)

# lamda = 3
# mu = 6
# chi = 40  # bps
# gamma = 50  # m/s
# queuing_delay = 1 / (lamda - mu)
# transmission_delay = Kbit / chi