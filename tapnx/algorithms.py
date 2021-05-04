""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx
import time
from . import utils_graph
from numba import njit, jit

from . import utils_graph


def _edge_func_np(x,a,b,c,n,d,lam,max_d,max_tt):
    return a*(1+b*(x/c)*(x/c)*(x/c)*(x/c))

def _edge_func_derivative_np(x,a,b,c,n,d,lam,max_d,max_tt):
    return (a*b*n*x*x*x)/(c*c*c*c)

def objective(x,a,b,c,n):
    return a*x*(1 + (b/(n+1))*(x/c)**n) 

def link_based_method(
    G, 
    method, 
    aec_gap_tol=10**-4, 
    max_iter=None, 
    line_search_tol=10**-3, 
    collect_data=False
):
    print('Solving')
    
    # Iteration
    #-----------
    # For every OD pair
    # Step 1: Generate an intial solution x
    # Step 2: Generate a target solution x*
    #   * SA - AON
    #   * FW - AON
    #   * CFW - Convex combination of old target y and aon y_aon using alpha (formula)
    # Step 3: Update the current solution x <- lx* +(1-l)x for l in [0,1]
    #   * SA - lam = 1/i
    #   * FW - line search that minimises objective
    #   * CFW - line search that minimises objective
    # Step 4: Calculate the travel times
    # Step 5: If convergence met (average excess cost AEC), stop. Otherwise return to Step 2

    shortest_paths = dict()

    no_edges = G.graph['no_edges']
    x = np.zeros(no_edges,dtype="float64") 
    a = utils_graph.get_np_array_from_edge_attribute(G, 'a')
    b = utils_graph.get_np_array_from_edge_attribute(G, 'b')
    cap = utils_graph.get_np_array_from_edge_attribute(G, 'c')
    n = utils_graph.get_np_array_from_edge_attribute(G, 'n')

    trips = G.graph['trips']

    # Update to store shortest paths that have been used 
    data = {'AEC':[], 'relative_gap':[], 'x':[], 'weight':[], 'objective':[], 'no_paths':[]}
    
    i = 0
    while True:
        weight = _edge_func_np(x,a,b,cap,n)
        utils_graph.update_edge_attribute(G, 'weight', weight)
        y = np.zeros(no_edges,dtype="float64")
        print('iteration {}...............'.format(i))
        # iterate through the Origin/Destination pairs
        for origin, destinations in trips.items():
            # Create a copy of the graph and remove the out edges that are connected to zones based on whether they are 
            # prohibited to be thru nodes
            H = G.copy()
            if G.graph['first_thru_node'] > 1:
                print('here')
                zones_prohibited = list(range(1,origin))
                zones_prohibited += list(range(origin+1, G.graph['first_thru_node']))
                out_edges_for_removal = G.out_edges(zones_prohibited)
                H.remove_edges_from(out_edges_for_removal)

            # calculate shortest paths, networkx does not (check) have an option for a list of targets
            lengths, all_paths = nx.single_source_dijkstra(H, source=origin, target=None, weight='weight')
            
            # iterate through the destinations for an origin
            for destination in destinations:
                demand = trips[origin][destination]
                # if there are positive trips
                if not ((demand == 0) or (demand == np.nan)):
                    #print(destination)
                    #print(all_paths)
                    shortest_path = all_paths[int(destination)]
                    shortest_path_length = lengths[int(destination)]
                    shortest_paths[(origin,destination)] = {'path': shortest_path, 'path_length': shortest_path_length}
                    path_edges = utils_graph.edges_from_path(shortest_path)
                    # refer to previous work or look for possible smarter way to update edges
                    path_edges = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]
                    y[path_edges] += demand
        
        
        AEC  = _average_excess_cost(trips, shortest_paths, x, weight)
        rel_gap = _relative_gap(trips, shortest_paths, x, weight)
        # store the data
        if collect_data and i > 0:                    
            print('i: {}'.format(i))
            print('AEC: {}'.format(AEC))
            print('rel gap: {}'.format(rel_gap))
            data['AEC'].append(AEC)
            data['relative_gap'].append(rel_gap)
            data['x'].append(x)
            data['weight'].append(weight)
            data['objective'].append(objective(x,a,b,cap,n))


        # convergence tests
        if AEC < aec_gap_tol  and i > 1:
            break

        if max_iter:
            if i > max_iter:
                break
        
        # generate new solution x based on previous solution x and new target y
        lam = 1
        if not i == 0:
            # Step 3: Update the current solution x <- lx* +(1-l)x for l in [0,1]
            if method == 'successive_averages':
                lam = 1/(i)
            else:
                lam = _line_search_fw(x, y, a, b, cap, n, tol=line_search_tol)
            
        # convex combination of old solution and target solution
        x = lam*y + (1-lam)*x
        
        
        i+=1
        
    return G, data

# need to update docstrings and kwgs (clean up)
def frank_wolfe(G, **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, data = link_based_method(G, 'frank_wolfe', **kwargs)
    return G, data

def _line_search_fw(x, y, a, b, c, n, tol=0.01):
    p = 0
    q = 1
    while True:
        alpha = (p+q)/2.0

        # dz/d_alpha derivative
        D_alpha = sum((y-x)*_edge_func_np(x + alpha*(y-x),a,b,c,n))
        
        if D_alpha <= 0:
            p = alpha
        else:
            q = alpha
        if q-p < tol:
            break
        
    return (p+q)/2

def successive_averages(G,  **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, data = link_based_method(G, 'successive_averages',  **kwargs)
    return G, data

def gradient_projection(G, 
    aec_gap_tol=10**-4, 
    max_iter=None, 
    collect_data=False,
    alpha = 1,
    edge_func = _edge_func_np,
    edge_func_derivative = _edge_func_derivative_np,
    d=False,
    lam=0,
    max_d=1,
    max_tt=1):

    # dictionary to store data about the instance the method was run
    data = {'AEC':[], 'relative_gap':[], 'x':[], 'weight':[], 'objective':[], 'no_paths':[]}

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
    t = edge_func(x,a,b,cap,n,d,lam,max_d,max_tt)
    trips = G.graph['trips']

    #dictionary to store shortest paths, indexed by (origin, destination)
    shortest_paths = dict()

    # dictionary to store paths for an (origin, destination)
    # e.g. {(origin, destination):{{'paths':[1,2,3] 'cost', 4, 'flow',1}} stores a path [1,2,3] with length 4 and flow 1
    od_paths = defaultdict(lambda: {'paths':[], 'D': [], 'h':[] })
    i = 0
    while True:
        print('Iteration i = {}--------------------\n'.format(i))
        
        # store the flow and weight at the start of the iteration (used for AEC)
        x_aec = np.copy(x)
        t_aec = np.copy(t)
        utils_graph.update_edge_attribute(G, 'weight',t)
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
                    
                    # if this is the first iteration, then assign all demand to shortest path
                    if i == 0:
                        od_paths[(origin, int(destination))]['h'][0] = demand

                    # get the edge/path matrix D and the path flow vector h
                    D = np.array(od_paths[(origin, int(destination))]['D']).T
                    h = np.array(od_paths[(origin, int(destination))]['h'])
                    
                    # if there is more than one path then attempt to shift flow
                    if len(h) > 1:
                        
                        # get path costs
                        c = np.dot(np.transpose(D),t)
                        print((origin, int(destination)))
                        print(c)
                        # get cheapest path id
                        sp_id = np.argmin(c)

                        # compute the gradient vector
                        g = c - c[sp_id]
                        
                        # get all non common links for each of the paths
                        non_common_links = np.abs(D - D[:,[sp_id]])
                        # compute the approximation of the hessian using the non_common_links
                        H = np.dot(np.transpose(non_common_links), t_prime)

                        # set this to something other than 0 to avoid division by 0, note that g[sp_id]=0, so valid
                        H[sp_id] = 1

                        # update h, not that the update for the shortest path will be 0
                        h_prime = np.maximum(np.zeros(len(h)), h - alpha*g/H)
                        
                        # compute the new flow on the shortest path based on the rest of the flows
                        h_prime[sp_id] += demand - np.sum(h_prime)
                        
                        # add new flow and subtract previous flow
                        x += np.dot(D,h_prime-h)
                        
                        # store the updated flow
                        od_paths[(origin, int(destination))]['h'] = h_prime.tolist()
                        
                        # update the travel times and the first derivative
                        t = edge_func(x,a,b,cap,n,d,lam,max_d,max_tt)
                        t_prime = edge_func_derivative(x,a,b,cap,n,d,lam,max_d,max_tt)
                                                
                    else:
                        # if first iteration then induce flow onto the edges x
                        # note that there is no need to anything if we have only one path and this is not the first iteration
                        if i == 0:
                            x += np.dot(D,h)
                            t = edge_func(x,a,b,cap,n,d,lam,max_d,max_tt)
                            t_prime = edge_func_derivative(x,a,b,cap,n,d,lam,max_d,max_tt)
                        
                        
        # compute average excess cost, based on travel times at the time of computing shortest paths.
        AEC  = _average_excess_cost(trips, shortest_paths, x_aec, t_aec)
        rel_gap = _relative_gap(trips, shortest_paths, x_aec, t_aec)
        
        
        if i > 0:
            print('AEC: {}'.format(AEC))
        
        # convergence tests
        #if AEC < aec_gap_tol and i > 0:
        #    break

        if max_iter:
            if i > max_iter:
                break

        # collect data about the iterations of the method
        if collect_data and i > 0:
            data['AEC'].append(AEC)
            data['relative_gap'].append(rel_gap)
            data['x'].append(x)
            data['weight'].append(t)
            data['objective'].append(objective(x,a,b,cap,n))
            data['no_paths'].append(get_D_matrix(od_paths).shape[1])
        i+=1
        #print(get_D_matrix(od_paths).shape)

    return G, data

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


