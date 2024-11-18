import pulp
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


solvers = ['GHPP', 'ML', 'EXP', 'PASSIVE']
stats = pkl.load(open("stats.pkl", "rb"))
plt.style.use('ggplot')

######################## COMPARING VARIOUS ALGORITHMS ##############################

COSTS = np.zeros((4,4,3))
DELAYS = np.zeros((4,4,2))

for prob in range(4):
    for i in range(4):
        COSTS[prob][i][0] = stats[prob][i]['total_cost']
        COSTS[prob][i][1] = stats[prob][i]['air_cost']
        COSTS[prob][i][2] = stats[prob][i]['ground_cost']

        DELAYS[prob][i][0] = stats[prob][i]['air_delay']
        DELAYS[prob][i][1] = stats[prob][i]['ground_delay']
    

    
def comp_costs(p):
    total = COSTS[p][:,0]
    air = COSTS[p][:,1]
    ground = COSTS[p][:,2]

    print(total)
    print(air)
    print(ground)

    plt.figure(figsize=(10,5))
    plt.bar(solvers, ground, width=0.4, label='Average Ground Cost')
    plt.bar(solvers, air, width=0.4, label='Average Air Cost', bottom=ground)
    plt.legend()
    
    plt.xticks(range(len(solvers)), solvers)  # Set x tick labels

    plt.xlabel('Algorithms')
    plt.ylabel('Average Cost')
    plt.show()

def comp_delays(p):
    air = DELAYS[p][:,0]
    ground = DELAYS[p][:,1]

    print(air)
    print(ground)

    plt.figure(figsize=(10,5))
    plt.bar(solvers, ground, width=0.4, label='Average Ground Delay')
    plt.bar(solvers, air, width=0.4, label='Average Air Delay', bottom=ground)
    plt.legend()
    
    plt.xticks(range(len(solvers)), solvers)  # Set x tick labels

    plt.xlabel('Algorithms')
    plt.ylabel('Average Delay (15 min periods)')
    plt.show()


def comp_classes(p):
    ground = stats[p][0]['ground_cost_class']

    print(ground)

    plt.figure(figsize=(10,5))
    plt.bar(range(3), ground, width=0.4, label='Average Ground Cost')
    plt.legend()
    
    plt.xticks(range(3), ["Class 1", "Class 2", "Class 3"])  # Set x tick labels

    plt.xlabel('Algorithms')
    plt.ylabel('Average Cost')
    plt.show()

comp_costs(0)
comp_costs(1)
comp_costs(2)
comp_costs(3)


comp_delays(0)
comp_delays(1)
comp_delays(2)
comp_delays(3)


comp_classes(0)