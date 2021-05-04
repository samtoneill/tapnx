""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx
import time
from . import utils_graph
from numba import njit, jit
from scipy.optimize import minimize, optimize, LinearConstraint

from . import utils_graph


#def _edge_func_np(x,a,b,c,n,d,lam,max_d,max_tt):
def _edge_func_np(x,a,b,c,n):
    return a*(1+b*(x/c)*(x/c)*(x/c)*(x/c))

def _edge_func_derivative_np(x,a,b,c,n,d,lam,max_d,max_tt):
    return (a*b*n*x*x*x)/(c*c*c*c)

def objective(x,a,b,c,n):
    return a*x*(1 + (b/(n+1))*(x/c)**n) 

def beckmann(h,D,a,b,c,n):
    x = np.dot(D,h)
    t = a*x*(1 + (b/(n+1))*(x/c)**n)
    return np.sum(t) 

def scipy_optimise_column_gen(
    G,
    edge_func = _edge_func_np,
    d=False,
    lam=0,
    max_d=1,
    max_tt=1):

    # compute demand vector

    # while convergence not met
        # for all (o,d) pairs
            # get shortest paths
            # if shortest path not already in paths
                # add

        # compute D matrix 
        # compute h vector (path)
        # compute A matrix (rows are paths relating to demand vector)


        # solve nonlin via scipy optimise

        # let F be the function to minimise
        # x = Dh

        # min F(Dh)
        # s.t 
        #     Ah  = d
        #      h >= 0

    # return x, h, F

    no_edges = G.graph['no_edges']

    # edge attributes used in the calculation of edge travel times
    a = utils_graph.get_np_array_from_edge_attribute(G, 'a')
    b = utils_graph.get_np_array_from_edge_attribute(G, 'b')
    cap = utils_graph.get_np_array_from_edge_attribute(G, 'c')
    n = utils_graph.get_np_array_from_edge_attribute(G, 'n')
    if d:
        d = utils_graph.get_np_array_from_edge_attribute(G, 'd')
    else:
        d = 1

    # initialise edge flow x and edge travel time t
    x = np.zeros(no_edges,dtype="float64") 
    t = edge_func(x,a,b,cap,n)
    trips = G.graph['trips']

    #dictionary to store shortest paths, indexed by (origin, destination)
    shortest_paths = dict()

    # dictionary to store paths for an (origin, destination)
    # e.g. {(origin, destination):{{'paths':[1,2,3] 'cost', 4, 'flow',1}} stores a path [1,2,3] with length 4 and flow 1
    od_paths = defaultdict(lambda: {'paths':[], 'D': [], 'h':[] })
    no_paths = 0
    no_od_pairs = 0
    demands = []
    i = 0

    for origin, destinations in trips.items():

        # Create a copy of the graph and remove the out edges that are connected to zones based on whether they are 
        # prohibited to be thru nodes

        J = G.copy()
        if G.graph['first_thru_node'] > 1:
            print('here')
            print(G.graph['first_thru_node'] )
            zones_prohibited = list(range(1,origin))
            zones_prohibited += list(range(origin+1, G.graph['first_thru_node']))
            out_edges_for_removal = G.out_edges(zones_prohibited)
            J.remove_edges_from(out_edges_for_removal)
            #print(zones_prohibited)

        # calculate shortest paths, networkx does not (check) have an option for a list of targets
        lengths, all_paths = nx.single_source_dijkstra(J, source=origin, target=None, weight='weight')
        # iterate through the destinations for an origin
        for destination in destinations:
            # calculate shortest paths, networkx does not (check) have an option for a list of targets
            
            # get demand associated with origin/destination
            demand = trips[origin][destination]
            # if there are positive trips (otherwise no need to do anything)
            if not ((demand == 0) or (demand == np.nan)):
                # get the shortest path and its length for the origin/destination
                no_od_pairs += 1
                demands.append(demand)
                shortest_path = all_paths[int(destination)]
                shortest_path_length = lengths[int(destination)]
                # get the paths for the 
                paths = od_paths[(origin, int(destination))]['paths']

                # overwrite the shortest path
                shortest_paths[(origin,destination)] = {'path': shortest_path, 'path_length': shortest_path_length}

                # if the shortest path is not yet in the paths, add it
                if not tuple(shortest_path) in paths:
                    
                    path_edges = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]
                    path_vector = np.zeros(no_edges,dtype="int32")
                    path_vector[path_edges] = 1
                    
                    od_paths[(origin, int(destination))]['D'].append(path_vector)
                    od_paths[(origin, int(destination))]['h'].append(0)
                    od_paths[(origin, int(destination))]['paths'].append(tuple(shortest_path))
                    no_paths += 1
                # if this is the first iteration, then assign all demand to shortest path
                if i == 0:
                    od_paths[(origin, int(destination))]['h'][0] = demand

    demands = np.array(demands)
    
    i = 0
    while True:
        i+=1
        print('Iteration i = {}--------------------\n'.format(i))

        A = get_A_matrix(od_paths, no_paths, no_od_pairs)
        D = get_D_matrix(od_paths) 
        h = get_h_vector(od_paths)
        print(D.shape)
        print(h.shape)
        print(A.shape)
        print(demands.shape)    
        h0 = np.ones(len(h))
        hb = (0,np.inf)
        bnds = tuple([hb for _ in range(len(h))])
        demand_cons = LinearConstraint(A, demands, demands)
        sol = minimize(beckmann, h0, bounds=bnds, constraints=demand_cons,args=(D,a,b,cap,n))
        h = sol.x
        x = np.dot(D,h)
        t = _edge_func_np(x,a,b,cap,n)   
        print(x)
        
        
        utils_graph.update_edge_attribute(J, 'weight',t)

        no_new_paths = 0
        for origin, destinations in trips.items():

            # calculate shortest paths, networkx does not (check) have an option for a list of targets
            lengths, all_paths = nx.single_source_dijkstra(J, source=origin, target=None, weight='weight')
            # iterate through the destinations for an origin
            for destination in destinations:
                # calculate shortest paths, networkx does not (check) have an option for a list of targets
                
                # get demand associated with origin/destination
                demand = trips[origin][destination]
                
                # if there are positive trips (otherwise no need to do anything)
                if not ((demand == 0) or (demand == np.nan)):
                    # get the shortest path and its length for the origin/destination
                    shortest_path = all_paths[int(destination)]
                    shortest_path_length = lengths[int(destination)]
                    # get the paths for the 
                    paths = od_paths[(origin, int(destination))]['paths']

                    # overwrite the shortest path
                    shortest_paths[(origin,destination)] = {'path': shortest_path, 'path_length': shortest_path_length}

                    # if the shortest path is not yet in the paths, add it
                    if not tuple(shortest_path) in paths:
                        no_new_paths += 1
                        path_edges = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]
                        path_vector = np.zeros(no_edges,dtype="int32")
                        path_vector[path_edges] = 1
                        
                        od_paths[(origin, int(destination))]['D'].append(path_vector)
                        od_paths[(origin, int(destination))]['h'].append(0)
                        od_paths[(origin, int(destination))]['paths'].append(tuple(shortest_path))
                        no_paths += 1

        if no_new_paths == 0:
            break
        print(no_new_paths)
        time.sleep(1)
                


def total_system_travel_time(x, weight):
    return np.dot(x,weight)

def _all_demand_on_fastest_paths(trips, shortest_paths):
    
    k = np.array([value['path_length'] for key, value in shortest_paths.items()])
    
    d = np.array([trips[key[0]][key[1]] for key, value in shortest_paths.items()])
    #print([(key,value) for key, value in G.graph['sp'].items()])
    #time.sleep(2)
    return np.dot(k,d)

def _relative_gap(trips, shortest_paths, x, weight):
    return (total_system_travel_time(x, weight)/_all_demand_on_fastest_paths(trips, shortest_paths))-1

def _average_excess_cost(trips, shortest_paths, x, weight):
    d =  np.array([trips[key[0]][key[1]] for key, value in shortest_paths.items()])
    
    return (total_system_travel_time(x, weight)-_all_demand_on_fastest_paths(trips, shortest_paths))/np.sum(d)

def get_D_matrix(od_paths):
    np_arrays = tuple([value['D'] for key, value in od_paths.items()])
    #flatten list
    np_arrays = [item for sublist in np_arrays for item in sublist]
    return np.column_stack(np_arrays)

def get_h_vector(od_paths):
    paths = tuple([value['h'] for key, value in od_paths.items()])
    
    #flatten list
    paths = [item for sublist in paths for item in sublist]
    return np.array(paths)

def get_A_matrix(od_paths, no_paths, no_od_pairs):
    
    A = np.zeros((no_od_pairs, no_paths))

    start_index = 0
    for indx, (key, value) in enumerate(od_paths.items()):
        no_od_paths = len(value['paths'])
        for i in range(start_index,start_index+no_od_paths):
            A[indx,i] = 1
        start_index = start_index + no_od_paths
    return A
    
