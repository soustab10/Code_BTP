import numpy as np

np.random.seed()

xm = 100
ym = 100
zm = 100

# x y z Coordinates of the Sink
sink = {'x': 0.5 * xm, 'y': 0.5 * ym, 'z': 0.5 * zm}

# Number of Nodes in the field
n = 100

# Optimal Election Probability of a node to become cluster head
p = 0.05

# Energy Model (all values in Joules)
# Initial Energy
Eo = 0.5
# Eelec=Etx=Erx
ETX = 50 * 0.000000001
ERX = 50 * 0.000000001
# Transmit Amplifier types
Efs = 10 * 0.000000000001
Emp = 0.0013 * 0.000000000001
# Data Aggregation Energy
EDA = 5 * 0.000000001

# Values for Heterogeneity
# Percentage of nodes that are advanced
m = 0.1
# \alpha
a = 1

# Maximum number of rounds
rmax = 1

# Computation of do/
do = np.sqrt(Efs / Emp)
packets_TO_BS = 0
np.set_printoptions(precision=15)

alpha = 4
beta = 2
gama = 2
delta = 4
Elec = 50 * 0.000000001  # Eelec = 50nJ/bit energy transfer and receive
Efs = 10 * 0.000000000001  # energy free space
Emp = 0.0013 * 0.000000000001  # energy multi-path
Kbit = 2000  # size
CH_Kbit = 200  # Adv. advertisement msg is 25 bytes long i.e. 200 bits
Eda = 5 * 0.000000001  # data aggregation nj/bit/signal
MaxInterval = 3  # TIME INTERVAL

oldnt = np.zeros(n + 1)
threshold = 35
Eresx = np.zeros(n + 1)
Eresy = np.zeros(n + 1)
summ = np.zeros(n + 1)
sumation = 0
permon = np.full(n + 1, 0.0001)
neighbour = np.zeros(n + 1)
pr = np.zeros(n + 1)

SinkID = n + 1
InitialEnergy = 0.5
Etx = 0
Erx = 0
SUM_Total_Energy = 0
Threshold = 0
I = 0
dpermon = 0
roo = 0.02
Rrout = 0
Rounds = 0
d0 = np.sqrt(Efs / Emp)
actual_rounds = 0
DEAD = np.zeros(rmax + 1)
DEAD_N = np.zeros(rmax + 1)
DEAD_A = np.zeros(rmax + 1)

lamda = 3
mu = 6
chi = 40  # bps
gamma = 50  # m/s
queuing_delay = 1 / (lamda - mu)
transmission_delay = Kbit / chi

# Creation of the random Sensor Network
S = []

for i in range(1, n + 1):
    S.append({'xd': np.random.rand() * xm,
              'yd': np.random.rand() * ym,
              'zd': np.random.rand() * zm,
              'G': 0,
              'type': 'N',
              'E': Eo,
              'ENERGY': 0})

S.append({'xd': sink['x'], 'yd': sink['y'], 'zd': sink['z'], 'type': 'S'})

for i in range(1, n + 1):
    temp_rnd0 = i
    if temp_rnd0 >= m * n + 1:
        S[i - 1]['E'] = Eo
    if temp_rnd0 < m * n + 1:
        S[i - 1]['E'] = Eo * (1 + a)

STATISTICS = {'DEAD': np.zeros(rmax + 1), 'CLUSTERHEADS': np.zeros(rmax + 1)}

for r in range(0, rmax + 1):
    if r % int(1 / p) == 0:
        for i in range(n):
            S[i]['G'] = 0

    dead = 0
    dead_a = 0
    dead_n = 0
    packets_TO_BS = 0
    packets_TO_CH = 0
    PACKETS_TO_BS[r] = 0
    PACKETS_TO_CH[r] = 0

    for i in range(n):
        if S[i]['E'] <= 0:
            dead += 1
            if S[i]['ENERGY'] == 1:
                dead_a += 1
            if S[i]['ENERGY'] == 0:
                dead_n += 1

    countCHs = 0
    cluster = 1

    for i in range(n):
        if S[i]['E'] > 0:
            temp_rand = np.random.rand()
            if S[i]['G'] <= 0:
                if temp_rand <= (p / (1 - p * (r % int(1 / p)))):
                    countCHs += 1
                    packets_TO_BS += 1
                    PACKETS_TO_BS[r] = packets_TO_BS
                    S[i]['type'] = 'C'
                    S[i]['G'] = int(1 / p) - 1
                    cluster += 1

                    distance = np.sqrt((S[i]['xd'] - S[n]['xd']) ** 2 +
                                       (S[i]['yd'] - S[n]['yd']) ** 2 +
                                       (S[i]['zd'] - S[n]['zd']) ** 2)

                    C = {'xd': S[i]['xd'],
                         'yd': S[i]['yd'],
                         'zd': S[i]['zd'],
                         'distance': distance,
                         'id': i + 1}

                    clusterinfo = np.zeros(n + 1)
                    clusterinfo[i + 1] = 1

                    S[i]['E'] -= (ETX + EDA) * (4000) + Emp * 4000 * (distance ** 4) if distance > do else (
                            ETX + EDA) * (4000) + Efs * 4000 * (distance ** 2)

    STATISTICS['DEAD'][r] = dead
    DEAD[r] = dead
    DEAD_N[r] = dead_n
    DEAD_A[r] = dead_a

    if dead == 1 and flag_first_dead == 0:
        first_dead = r
        flag_first_dead = 1

    countCHs = 0
    cluster = 1
    clusterinfo = np.zeros(n + 1)

    for i in range(n):
        if S[i]['E'] > 0:
            temp_rand = np.random.rand
            if (S[i]['G']) <= 0:

                # Election of Cluster Heads
                if temp_rand <= (p / (1 - p * (r % int(1 / p)))):
                    countCHs += 1
                    packets_TO_BS += 1
                    PACKETS_TO_BS[r] = packets_TO_BS
                    clusterinfo[i + 1] = 1
                    S[i]['type'] = 'C'
                    S[i]['G'] = int(1 / p) - 1
                    C = {'xd': S[i]['xd'],
                         'yd': S[i]['yd'],
                         'zd': S[i]['zd'],
                         'distance': distance,
                         'id': i + 1}

                    S[i]['E'] -= (ETX + EDA) * (4000) + Emp * 4000 * (distance ** 4) if distance > do else (
                            ETX + EDA) * (4000) + Efs * 4000 * (distance ** 2)

    STATISTICS['CLUSTERHEADS'][r] = cluster - 1

    for i in range(n):
        if S[i]['type'] == 'N' and S[i]['E'] > 0:
            if cluster - 1 >= 1:
                min_dis = np.sqrt((S[i]['xd'] - S[n]['xd']) ** 2 +
                                  (S[i]['yd'] - S[n]['yd']) ** 2 +
                                  (S[i]['zd'] - S[n]['zd']) ** 2)
                min_dis_cluster = 1

                for c in range(1, cluster):
                    temp = min(min_dis, np.sqrt((S[i]['xd'] - C[c]['xd']) ** 2 +
                                                (S[i]['yd'] - C[c]['yd']) ** 2 +
                                                (S[i]['zd'] - C[c]['zd']) ** 2))
                    if temp < min_dis:
                        min_dis = temp
                        min_dis_cluster = c

                if min_dis > 0:
                    S[C[min_dis_cluster]['id'] - 1]['E'] -= (ERX + EDA) * 4000
                    PACKETS_TO_CH[r] = n - dead - cluster + 1

                S[i]['min_dis'] = min_dis
                S[i]['min_dis_cluster'] = min_dis_cluster

    countCHs
    rcountCHs += countCHs

clusterinfo[101] = 1

for i in range(n + 1):
    if clusterinfo[i] == 1:
        infnode = i
        break

dist_direct = np.zeros((n + 1, n + 1))

for i in range(n + 1):
    if clusterinfo[i] == 1:
        continue
    for j in range(n + 1):
        if clusterinfo[j] == 1:
            dist_direct[i, j] = np.sqrt((S[i - 1]['xd'] - S[j - 1]['xd']) ** 2 +
                                        (S[i - 1]['yd'] - S[j - 1]['yd']) ** 2 +
                                        (S[i - 1]['zd'] - S[j - 1]['zd']) ** 2)

r = 1
neighbour = np.zeros(n + 1)

while True:
    for j in range(n + 1):
        if dist_direct[infnode, j] <= threshold and dist_direct[infnode, j] != 0:
            neighbour[j] = j

    for h in range(n + 1):
        if neighbour[h] == oldnt[h]:
            neighbour[h] = 0

    if r != 1:
        neighbour[Rrout] = 0

    if neighbour.all() == 0:
        threshold = 60
        continue

    if neighbour[n] == n:
        print("Reached the sink Node")
        I = n
        print(I)
        S[infnode - 1]['E'] -= Kbit * Elec + Efs * (dist_direct[infnode, n] ** 2)
        Rrout[r] = infnode
        break

    for k in range(n + 1):
        if neighbour[k] == 0:
            continue
        Etx[infnode] = Kbit * Elec + Efs * (dist_direct[infnode, k] ** 2)
        Erx[k] = Kbit * Elec
        Eresx[infnode] = S[infnode - 1]['E'] - Etx[infnode]
        Eresy[k] = S[k - 1]['E'] - Erx[k]

        summ[k] = (dist_direct[infnode, k] ** alpha) * (permon[k] ** beta) * (Eresx[infnode] ** gama) * (
                Eresy[k] ** delta)

        sumation = np.sum(summ)

    for k in range(n + 1):
        if neighbour[k] == 0:
            continue
        Etx[infnode] = Kbit * Elec + Efs * (dist_direct[infnode, k] ** 2)
        Erx[k] = Kbit * Elec
        Eresx[infnode] = S[infnode - 1]['E'] - Etx[infnode]
        Eresy[k] = S[k - 1]['E'] - Erx[k]

        pr[k] = (dist_direct[infnode, k] ** alpha) * (permon[k] ** beta) * (Eresx[infnode] ** gama) * (
                Eresy[k] ** delta) / sumation

    for k in range(n + 1):
        M, I = max(pr), np.argmax(pr)
        if M != 0:
            if S[i - 1]['E'] < 0.2:
                pr[I] = 0
            else:
                continue

    print(pr)

    for k in range(n + 1):
        if pr[k] > 0:
            S[k - 1]['E'] = Eresy[k]
            S[infnode - 1]['E'] = Eresx[infnode]
            oldnt = neighbour.copy()

            if S[I - 1]['E'] <= 0:
                StateNode1[I] = 0
            if S[infnode - 1]['E'] <= 0:
                StateNode1[infnode] = 0

            Rrout[r] = infnode
            infnode = I
            r += 1
