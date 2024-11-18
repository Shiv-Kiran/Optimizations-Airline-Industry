import pulp
import numpy as np
import matplotlib.pyplot as plt


"""
GHPP With multiple aircraft classes

Input:
    T -> Number of time periods.
    Q -> Number of capacity scenarios.
    K -> Number of aircraft classes.
    C -> Q x T+1 matrix of airport capacity scenarios.
    P -> Q x 1 vector of probabilities of capacity scenarios.
    land -> K x T matrix of number of flights of each class landing at each time period.
    G -> K x 1 vector of ground costs of each aircraft class.
    A -> Air cost per unit delay.
"""


def GHPP_solver(T, Q, K,  C, P, land, G, A):
    prob = pulp.LpProblem("Airport", pulp.LpMinimize)


    #### decision variables
    # xijk = num of flights of type k scheuled to land in period of i but land at period j. 0 <= i < T, i <= j <= T
    X = {}
    for i in range(T):
        for j in range(i, T+1):
            for k in range(K):
                X[(i,j,k)] = pulp.LpVariable("x_{}_{}_{}".format(i,j,k), lowBound=0, upBound=50, cat='Integer')


    # y[i][q] = num of flights bearing air delay at period i. 0 <= i < T, in case of schedule q.
    Y = {}
    for i in range(T):
        for q in range(Q):
            Y[(i,q)] = pulp.LpVariable("y_{}_{}".format(i,q), lowBound=0, upBound=300,cat='Integer')


    ## Constraints
    for i in range(T):
        for k in range(K):
            prob += pulp.lpSum( X[(i,j,k)] for j in range(i, T+1) ) == land[k][i]

    for q in range(Q):
        for i in range(1,T):
            prob += Y[(i,q)] >= pulp.lpSum( X[(j,i,k)] for j in range(i+1) for k in range(K) ) + Y[(i-1,q)] - C[q][i]

        Y[(0, q)] >=  pulp.lpSum( X[(0,0,k)] for k in range(K) ) - C[q][0]


    ## Objective
    prob += A*pulp.lpSum(P[q]*Y[(i,q)] for i in range(T) for q in range(Q)) + pulp.lpSum( pulp.lpSum( X[(i,j,k)]*(j - i)*G[k]  for i in range(T) for j in range(i+1, T+1) ) for k in range(K) )



    prob.solve( pulp.PULP_CBC_CMD(msg=0) )


    res = {
        "status" : pulp.LpStatus[prob.status],
        "cost" : pulp.value(prob.objective),
        "X" : X,
        "Y" : Y
    }

    # print("Status:", pulp.LpStatus[prob.status])
    # print("The optimal cost is: {}".format(pulp.value(prob.objective)))

    return res


def most_likely_solver(T, Q, K,  C, P, land, G, A):

    p_max = 0
    i_max = -1
    for q in range(Q):
        if P[q] > p_max:
            p_max = P[q]
            i_max = q

    res = GHPP_solver(T, 1 , K,  C[i_max, :].reshape(1,-1), [1], land, G, A)

    return res



def expected_weather_solver(T, Q, K,  C, P, land, G, A):
    C_exp = np.zeros((1, T+1))
    for q in range(Q):
        C_exp[0, :] += P[q]*C[q, :]
    
    res = GHPP_solver(T, 1 , K,  C_exp, [1], land, G, A)
    return res


def passive_solver(T, Q, K,  C, P, land, G, A):
    prob = pulp.LpProblem("Airport", pulp.LpMinimize)


    #### decision variables
    # xijk = num of flights of type k scheuled to land in period of i but land at period j. 0 <= i < T, i <= j <= T
    X = {}
    for i in range(T):
        for j in range(i, T+1):
            for k in range(K):
                X[(i,j,k)] = pulp.LpVariable("x_{}_{}_{}".format(i,j,k), lowBound=0, upBound=50, cat='Integer')


    # y[i][q] = num of flights bearing air delay at period i. 0 <= i < T, in case of schedule q.
    Y = {}
    for i in range(T):
        for q in range(Q):
            Y[(i,q)] = pulp.LpVariable("y_{}_{}".format(i,q), lowBound=0, upBound=300,cat='Integer')


    ## Constraints
    for i in range(T):
        for k in range(K):
            prob += pulp.lpSum( X[(i,j,k)] for j in range(i, T+1) ) == land[k][i]
            prob += X[(i,i,k)] == land[k][i]

    

    for q in range(Q):
        for i in range(1,T):
            prob += Y[(i,q)] >= pulp.lpSum( X[(i,i,k)]  for k in range(K) ) + Y[(i-1,q)] - C[q][i]

        Y[(0, q)] >=  pulp.lpSum( X[(0,0,k)] for k in range(K) ) - C[q][0]


    ## Objective
    prob += A*pulp.lpSum(P[q]*Y[(i,q)] for i in range(T) for q in range(Q))



    prob.solve( pulp.PULP_CBC_CMD(msg=0) )


    res = {
        "status" : pulp.LpStatus[prob.status],
        "cost" : pulp.value(prob.objective),
        "X" : X,
        "Y" : Y
    }

    return res


def solve_for_Y(T, Q, K,  C, P, land, G, A, X):
    #### decision variables
    # y[i][q] = num of flights bearing air delay at period i. 0 <= i < T, in case of schedule q.
    Y = {}
    for i in range(T):
        for q in range(Q):
            Y[(i,q)] = 0

    for q in range(Q):
        queue = 0
        for i in range(T):
            queue += sum( X[(j,i,k)].varValue for j in range(i+1) for k in range(K) )
            Y[(i,q)] = max(0, queue - C[q][i])
            queue = Y[(i,q)]

    return Y




def analyse_stats( T, Q, K, C, P, land, G, A, X, Y):
    """
        Return These measures, given a schedule and a solution:
            1. Total cost  : Air, Ground
            2. Total delay : Air, Ground
            3. Cost and Delay per aircraft class

            All these are expectations over the weather w.r.t P
    """
    air_cost = 0
    ground_cost = np.zeros((K,))
    total_ground_cost = 0

    air_delay = 0
    ground_delay = np.zeros((K,))
    total_ground_delay = 0


    for i in range(T):
        for q in range(Q):
            air_delay += P[q]*Y[(i,q)]

    for i in range(T):
        for j in range(i+1, T+1):
            for k in range(K):
                cost = (j - i)*G[k]*X[(i,j,k)].varValue
                ground_cost[k] += cost

                delay = (j - i)*X[(i,j,k)].varValue
                ground_delay[k] += delay


    air_cost = A*air_delay
    total_ground_cost = sum( ground_cost[k] for k in range(K) )
    total_ground_delay = sum( ground_delay[k] for k in range(K) )
    total_cost = air_cost + total_ground_cost

    res = {
        "total_cost" : total_cost,
        "air_cost" : air_cost,
        "air_delay" : air_delay,
        "ground_cost" : total_ground_cost,
        "ground_delay" : total_ground_delay,
        "ground_cost_class" : ground_cost,
        "ground_delay_class" : ground_delay
    }

    return res



def solver( T, Q, K , C, P , land, G, A, type = "GHPP" ):
    if type == "GHPP":
        res = GHPP_solver(T, Q, K, C, P, land, G, A)
    elif type == "most_likely":
        res = most_likely_solver(T, Q, K, C, P, land, G, A)
    elif type == "expected":
        res = expected_weather_solver(T, Q, K, C, P, land, G, A)
    elif type == "passive":
        res = passive_solver(T, Q, K, C, P, land, G, A)
    else:
        print("Invalid type of solver")
        res = {}
        return res
    
    X = res['X']
    res['Y'] = solve_for_Y(T, Q, K, C, P, land, G, A, X)
    Y = res['Y']

    perf = analyse_stats(T, Q, K, C, P, land, G, A, X, Y)
    res[ 'perf' ] = perf

    return res