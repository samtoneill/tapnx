""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx

from . import utils_graph

def link_based_method(G, method, aec_gap_tolerance=10**-4, max_iter=None):
    
    G.graph['sp'] = defaultdict(lambda: None)
    x = np.zeros(len(G.edges()),dtype="float64")
    y = np.zeros(len(G.edges()),dtype="float64")
    
    i = 1
    while True:
        update_edge_attribute(G, 'x', x)
        update_edge_attribute(G, 'weight', edge_func(G,x))

        # iterate through the Origin/Destination pairs
        for key, values in G.graph['trips'].items():
            # iterate through the destinations for an origin
            for value in values:
                demand = G.graph['trips'][key][value]
                # if there are positive trips
                if not ((demand == 0) or (demand == np.nan)):
                    path = nx.dijkstra_path(G, int(key), int(value), weight='weight')
                    path_length = utils_graph.path_length(G, path)
                    G.graph['sp'][(key,value)] = {'path': path, 'path_length': path_length}
                    path_edges = utils_graph.edges_from_path(path)
                    # refer to previous work or look for possible smarter way to update edges
                    for u,v in path_edges:
                        edge_id = G[u][v]['id']
                        #print(edge_id)
                        y[edge_id] += demand
        
        #print(relative_gap(G))
        AEC  = average_excess_cost(G)
        rel_gap = relative_gap(G)
        print('AEC: {}'.format(AEC))
        print('rel gap: {}'.format(rel_gap))
        if AEC < aec_gap_tolerance  and i > 1:
            break
        
        if max_iter:
            if i > max_iter:
                break
        
        lam = 1
        if not i == 1:
            # Step 3: Update the current soluiton x <- lx* +(1-l)x for l in [0,1]
            if method == 'successive_averages':
                lam = 1/(i)
            elif method == 'frank_wolfe':
                lam = line_search(G, x, y, epsilon=0.005)
            
        # convex combination of solutions
        x = lam*y + (1-lam)*x

        # reset target edge to 0
        y = np.zeros(len(G.edges()),dtype="float64") 

        #print(x)

        i+=1
        
    print(get_np_array_from_edge_attribute(G, 'weight'))
    print(objective(G,x))
    print(G.edges(data=True))
    # Iteration
    #-----------
    # For every OD pair
    # Step 1: Generate an intial solution x
    # Step 2: Generate a target solution x*
    # Step 3: Update the current soluiton x <- lx* +(1-l)x for l in [0,1]
    # Step 4: Calculate the travel times
    # Step 5: If convergence met (relative gap), stop. Otherwise return to Step 2
    return True

def frank_wolfe(G, aec_gap_tolerance=10**-4):
    link_based_method(G, 'frank_wolfe', aec_gap_tolerance=aec_gap_tolerance)
    return G

def successive_averages(G, aec_gap_tolerance=1):
    link_based_method(G, 'successive_averages', aec_gap_tolerance=aec_gap_tolerance)
    return G

def total_system_travel_time(G):
    x = get_np_array_from_edge_attribute(G, 'x')
    t = get_np_array_from_edge_attribute(G, 'weight')
    return np.dot(x,t)

def all_demand_on_fastest_paths(G):
    k = np.array([value['path_length'] for key, value in G.graph['sp'].items()])
    d = np.array([G.graph['trips'][key[0]][key[1]] for key, value in G.graph['sp'].items()])
    return np.dot(k,d)

def relative_gap(G):
    return (total_system_travel_time(G)/all_demand_on_fastest_paths(G))-1

def average_excess_cost(G):
    d = d = np.array([G.graph['trips'][key[0]][key[1]] for key, value in G.graph['sp'].items()])
    return (total_system_travel_time(G)-all_demand_on_fastest_paths(G))/np.sum(d)
    
def line_search(G, x, y, epsilon=0.01):
    p = 0
    q = 1
    while True:
        alpha = (p+q)/2.0
        D_alpha = sum((y-x)*edge_func(G, x + alpha*(y-x)))
        
        if D_alpha <= 0:
            p = alpha
        else:
            q = alpha
        if q-p < epsilon:
            break
        
    return (p+q)/2

def update_edge_attribute(G, attr, vector):
    d = dict(zip(sorted(G.edges()), vector))
    nx.set_edge_attributes(G, d, attr)
    return G

def edge_func(G,x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return list(map(G.graph['edge_func'], u,v,d,x))

def get_np_array_from_edge_attribute(G, attr):
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, attr).items())])

def objective(G, x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return sum(list(map(G.graph['objective'], u,v,d,x)))