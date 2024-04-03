import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
# Data for dead nodes
nodes = [250, 500, 750, 1000]
dead_EECRAP = [85, 132, 153, 272]
dead_Mine_MOT = [42, 42, 44, 45]
dead_EEGNBR = [62, 143, 178, 231]
dead_DVOR = [65, 216, 337, 374]
dead_DBR = [146, 311, 322, 452]

plt.figure(figsize=(10, 6))

plt.plot(nodes, dead_EECRAP, marker='o', label='EECRAP')
plt.plot(nodes, dead_Mine_MOT, marker='o', label='My Algo')
plt.plot(nodes, dead_EEGNBR, marker='o', label='EEGNBR')
plt.plot(nodes, dead_DVOR, marker='o', label='DVOR')
plt.plot(nodes, dead_DBR, marker='o', label='DBR')

plt.title('Dead Nodes vs. Number of Total Nodes')
plt.xlabel('Number of Total Nodes')
plt.ylabel('Dead Nodes')
plt.grid(True)
plt.legend()
plt.xticks(nodes)

# Adding data points
for i in range(len(nodes)):
    plt.text(nodes[i], dead_EECRAP[i], f"{dead_EECRAP[i]}", ha='right', va='bottom')
    plt.text(nodes[i], dead_Mine_MOT[i], f"{dead_Mine_MOT[i]}", ha='right', va='bottom')
    plt.text(nodes[i], dead_EEGNBR[i], f"{dead_EEGNBR[i]}", ha='right', va='bottom')
    plt.text(nodes[i], dead_DVOR[i], f"{dead_DVOR[i]}", ha='right', va='bottom')
    plt.text(nodes[i], dead_DBR[i], f"{dead_DBR[i]}", ha='right', va='bottom')

plt.tight_layout()
plt.show()

# Data for total residual energy
total_energy_EECRAP = [3035.563508, 3726.321768, 7692.812484, 9019.935624]
total_energy_Mine_MOT = [2783.527526, 6174.695326, 9492.988209, 13026.09441]
total_energy_EEGNBR = [2246.143768, 4171.564257, 6503.443889, 8697.360925]
total_energy_DVOR = [2311.686216, 3687.166787, 5246.702038, 8273.825338]
total_energy_DBR = [832.6585948, 1686.690131, 3300.293471, 5916.703935]

plt.figure(figsize=(10, 6))

plt.plot(nodes, total_energy_EECRAP, marker='o', label='EECRAP ')
plt.plot(nodes, total_energy_Mine_MOT, marker='o', label='My Algo ')
plt.plot(nodes, total_energy_EEGNBR, marker='o', label='EEGNBR ')
plt.plot(nodes, total_energy_DVOR, marker='o', label='DVOR')
plt.plot(nodes, total_energy_DBR, marker='o', label='DBR')

plt.title('Total Residual Energy vs. Number of Total Nodes')
plt.xlabel('Number of Total Nodes')
plt.ylabel('Total Residual Energy')
plt.grid(True)
plt.legend()
plt.xticks(nodes)
plt.tight_layout()
plt.show()

# Data for end-to-end (E2E) delay
e2e_EECRAP = [0.723879922, 0.889010145, 0.754596213, 0.758335463]
e2e_Mine_MOT = [0.766660476, 0.786146338, 0.798253394, 0.714908327]
e2e_EEGNBR = [0.746802995, 0.796901964, 0.857915502, 0.767669789]
e2e_DVOR = [0.966092277, 1.403363703, 1.153397396, 1.332186072]
e2e_DBR = [0.770293242, 0.747163901, 0.742610273, 0.803706215]

plt.figure(figsize=(10, 6))

plt.plot(nodes, e2e_EECRAP, marker='o', label='EECRAP ')
plt.plot(nodes, e2e_Mine_MOT, marker='o', label='My Algo ')
plt.plot(nodes, e2e_EEGNBR, marker='o', label='EEGNBR ')
plt.plot(nodes, e2e_DVOR, marker='o', label='DVOR')
plt.plot(nodes, e2e_DBR, marker='o', label='DBR')

plt.title('End-to-End (E2E) Delay vs. Number of Total Nodes')
plt.xlabel('Number of Total Nodes')
plt.ylabel('End-to-End (E2E) Delay')
plt.grid(True)
plt.legend()
plt.xticks(nodes)
plt.tight_layout()
plt.show()
