""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx
import time
from . import utils_graph
from numba import njit, jit
from scipy.optimize import minimize, optimize, LinearConstraint
from gekko import GEKKO
from itertools import islice

from . import utils_graph


def k_shortest_paths(G, source, target, k, weight=None):
    return islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)

# due to the way Gekko returns the solution, flatten list is done differently
def edge_flows_for_sol(f,D,a,b,c,n):
  f_flat = [item[0] for sublist in f for item in sublist]
  return np.dot(D,np.array(f_flat))

def edge_flows(f,D,a,b,c,n):
  f_flat = [item for sublist in f for item in sublist]
  return np.dot(D,np.array(f_flat))

def edge_flows_no_zeros(f,D,a,b,c,n,m):
  f_flat = [item for sublist in f for item in sublist]
  x = []
  for i in range(D.shape[0]):
    xi = []
    for i,a in enumerate(list(D[i,:])):
      if a:
        xi.append(f_flat[i])
    x.append(m.sum(xi))

  return np.array(x)

def system_optimal(x,a,b,c,n,m):
  return m.sum(x*(a*(1+b*(x/c)**n)))

def weighted_system_optimal(x,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m):
    dist = d*x
    tt = x*(a*(1+b*(x/c)**n))
    return m.sum( lam*( (dist-min_d)/(max_d-min_d) ) + (1-lam)*( (tt-min_tt)/(max_tt-min_tt) ) )

def beckmann_sum(x,a,b,c,n,m):
  return m.sum(a*x*(1 + (b/(n+1))*(x/c)**n) )

def beckmann_edge(x,i,a,b,c,n,m):
  return a[i]*x*(1 + (b[i]/(n[i]+1))*(x/c[i])**n[i])


def edge_costs_for_sol(x,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m):
    dist = d*x
    tt = x*(a*(1+b*(x/c)**n))
    return lam*( (dist-min_d)/(max_d-min_d) ) + (1-lam)*( (tt-min_tt)/(max_tt-min_tt) )

def path_costs_for_sol(f,D,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m):
    x = edge_flows(f,D,a,b,c,n)
    return np.dot(np.transpose(D), edge_costs_for_sol(x,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m))

def edge_func(x,a,b,c,n):
    return a*(1+b*(x/c)*(x/c)*(x/c)*(x/c))

def gekko_optimise_column_gen(
    G,
    d=False,
    lam=0,
    min_max_type = 1,
    min_d=0,
    min_tt=0,
    max_d=1,
    max_tt=1,
    remote=False,
    initial_paths=1):

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
    c = utils_graph.get_np_array_from_edge_attribute(G, 'c')
    n = utils_graph.get_np_array_from_edge_attribute(G, 'n')
    if d:
        d = utils_graph.get_np_array_from_edge_attribute(G, 'd')
    else:
        d = 1

    # initialise edge flow x and edge travel time t
    x = np.zeros(no_edges,dtype="float64") 
    t = edge_func(x,a,b,c,n)
    trips = G.graph['trips']

    #dictionary to store shortest paths, indexed by (origin, destination)
    shortest_paths = dict()

    # dictionary to store paths for an (origin, destination)
    # e.g. {(origin, destination):{{'paths':[1,2,3] 'cost', 4, 'flow',1}} stores a path [1,2,3] with length 4 and flow 1
    od_paths = defaultdict(lambda: {'paths':[], 'D': [], 'h':[] })
    no_paths = 0

    demands = []
    od_ids = []
    i = 0
    od_id = 0
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
        #lengths, all_paths = nx.single_source_dijkstra(J, source=origin, target=None, weight='weight')
        # iterate through the destinations for an origin
        for destination in destinations:
            
            
            # calculate shortest paths, networkx does not (check) have an option for a list of targets
            
            # get demand associated with origin/destination
            demand = trips[origin][destination]
            # if there are positive trips (otherwise no need to do anything)
            if not ((demand == 0) or (demand == np.nan)):
                
                # get the shortest path and its length for the origin/destination
                od_ids.append(od_id)
                
                demands.append(demand)
                paths = k_shortest_paths(G, origin, destination, initial_paths)
                for path in paths:
                    path_edges = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(path)]
                    path_vector = np.zeros(no_edges,dtype="int32")
                    path_vector[path_edges] = 1
                    
                    od_paths[od_id]['D'].append(path_vector)
                    od_paths[od_id]['h'].append(0)
                    od_paths[od_id]['paths'].append(tuple(path))
                    no_paths += 1
                # if this is the first iteration, then assign all demand to shortest path

                print(od_paths[od_id]['paths'])
                od_id += 1

    # require 
    # od_ids list of od ids
    # od_paths_ids dictionary of j paths for key od k
    demands = np.array(demands)
    
    i = 0
    while True:
        i+=1
        print('Iteration i = {}--------------------\n'.format(i))
        m = GEKKO(remote=remote)  

        #Set global options
        m.options.IMODE = 3 #steady state optimization
        m.options.SOLVER = 3
        
        od_paths_ids = {}
        for od_index, value in od_paths.items():
            od_paths_ids[od_index] = [int(indx) for indx, path in enumerate(value['paths'])]
        
        D = get_D_matrix(od_paths) 
        
        f = [[m.Var(lb=0) for j in od_paths_ids[k]] for k in od_ids]

        eqs = m.Equations(m.sum([f[k][j] for j in od_paths_ids[k]]) == demands[k] for k in od_ids)

        x = edge_flows_no_zeros(f,D,a,b,c,n,m)
        
        obj = weighted_system_optimal(x,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m)
        #obj = beckmann_sum(x,a,b,c,n,m)
        
        m.Obj(min_max_type*obj)
        
        #Solve simulation
        #m.solve(disp = True)  
        m.solve()  

        x = edge_flows_for_sol(f,D,a,b,c,n)
        t = edge_costs_for_sol(x,a,b,c,n,d,lam,min_d,max_d,min_tt,max_tt,m)

        
        
        utils_graph.update_edge_attribute(J, 'weight',t)

        no_new_paths = 0
        od_id = 0
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
                    paths = od_paths[od_id]['paths']

                    # overwrite the shortest path
                    shortest_paths[(origin,destination)] = {'path': shortest_path, 'path_length': shortest_path_length}

                    # if the shortest path is not yet in the paths, add it
                    if not tuple(shortest_path) in paths:
                        no_new_paths += 1
                        path_edges = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]
                        path_vector = np.zeros(no_edges,dtype="int32")
                        path_vector[path_edges] = 1
                        
                        od_paths[od_id]['D'].append(path_vector)
                        od_paths[od_id]['h'].append(0)
                        od_paths[od_id]['paths'].append(tuple(shortest_path))
                        no_paths += 1

                    od_id += 1

        if no_new_paths == 0:
            break
    
    return edge_flows(f,D,a,b,c,n), path_costs_for_sol, x, edge_costs_for_sol, m.options.OBJFCNVAL
        
                


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
    
