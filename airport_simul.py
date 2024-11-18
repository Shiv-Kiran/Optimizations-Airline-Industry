import pulp
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from solver import *

np.random.seed(1)

# S O M E  C O N S T A N T S,   U T I L I T I E S
##############################################################################################################

T = 96      ## Number of time periods.
Q = 4       ## Number of capacity scenarios.
K = 3       ## Number of aircraft classes.


######################################
#Q capacity scenarios of airport capacity

C = np.zeros((Q, T+1)) + 10

C[1, 50 : 70] = 5

C[2, 20 : 35] = 6
C[2, 50 : 65] = 6

C[3,30:40] = 6
C[3,40:50] = 3
C[3,50:60] = 6



## call this function to visualize the capacity scenarios
def vis_capacity():
    for i in range(Q):
        plt.subplot(Q, 1, i+1)
        plt.bar(range(T+1), C[i])
        plt.ylabel("Airport Capacity")
        plt.title("Scenario {}".format(i))
    plt.show()


####################################
#  Generating a random schedule of N flights

def gen_rand_num():
    """
        To generate a distribution more skewed towards the center
    """
    x1 = np.random.randint(1, T)
    x2 = np.random.randint(1, T)

    skewness = 0.8

    if abs(x1 - T/2) > abs(x2 - T/2):
        swap = x1
        x1 = x2
        x2 = swap

    choice = np.random.choice([0, 1], p=[skewness, 1-skewness])
    if choice == 0:
        return x1
    else:
        return x2

def gen_schedule(multi):

    classes = K
    frac = [ 0.45, 0.45, 0.1 ]
    if not multi:
        classes = 1
        frac = [1]

    land = np.zeros((classes, T))
    N = int(T*8)                ## 80% of the MAX capacity

    for i in range(classes):
        for j in range( int(N*frac[i]) ):
            end = gen_rand_num()
            land[i, end ] += 1

    return land


## call this function to visualize the schedule
def vis_schedule(multi, land):
    if not multi:
        plt.bar(range(T), land[0])
        plt.title("Schedule of flights")
        plt.show()
    else:
        plt.bar(range(T), land[0], label="Class 1")
        plt.bar(range(T), land[1], bottom=land[0], label="Class 2")
        plt.bar(range(T), land[2], bottom=land[1] + land[0], label="Class 3")
        plt.legend()
        plt.title("Schedule of flights")
        plt.show()


####################################
# Delay costs : Ground and Air

GHDM_multi = [ 150, 200, 400]       ## costs of all 3 classes of planes.

ADC = 750      #### Air delay cost per period



#######################################################################################################################
## PERFORMANCE ANALYTICS



#### 4 probability profiles, each profile has a probability value to each of the 4 capacity scenarios
p1 = [1/Q for i in range(Q)]
p2 = [0.5, 0.2, 0.2, 0.1]
p3 = [0.1, 0.4, 0.3, 0.2]
p4 = [0.9, 0.1, 0, 0]

P = [p1, p2, p3, p4]

solvers = ['GHPP', 'most_likely', 'expected', 'passive']
NUM_ITERS = 5

stats = []

for prob in range(4):

    results = []

    for _ in solvers:
        results.append(
            {
                'total_cost' : 0,
                'air_cost' : 0,
                'air_delay' : 0,
                'ground_cost' : 0,
                'ground_delay' : 0,
                'ground_cost_class' : np.zeros((K,)),
                'ground_delay_class' : np.zeros((K,))
            }
        )


    for i in range(NUM_ITERS):
        sched = gen_schedule(True)

        for j in range( len(solvers) ):
            res = solver(T, Q, K, C, P[prob], sched, GHDM_multi, ADC, solvers[j])
            for key in results[j]:
                results[j][key] += res['perf'][key]



    for j in range( len(solvers) ):
        for key in results[j]:
            results[j][key] /= NUM_ITERS

    stats.append(results)


pkl.dump(stats, open("stats.pkl", "wb"))

