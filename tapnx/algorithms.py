""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx
import time
from numba import njit, jit

from . import utils_graph

def link_based_method(
    G, 
    method, 
    aec_gap_tol=10**-4, 
    max_iter=None, 
    line_search_tol=10**-3, 
    collect_data=False, 
    edge_func_derivative=None
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

    G.graph['sp'] = defaultdict(lambda: None)
    x = np.zeros(len(G.edges()),dtype="float64")

    # Update to store shortest paths that have been used 
    data = {'AEC':[], 'relative_gap':[], 'x':[], 'weight':[], 'objective':[]}
    
    i = 1
    while True:
        _update_edge_attribute(G, 'x', x)
        _update_edge_attribute(G, 'weight', _edge_func(G,x, G.graph['edge_func']))
        y_aon = np.zeros(len(G.edges()),dtype="float64")
        #print('iteration {}...............'.format(i))
        # iterate through the Origin/Destination pairs
        for key, values in G.graph['trips'].items():
            lengths, paths = nx.single_source_dijkstra(G, source=key, target=None)
            # iterate through the destinations for an origin
            for value in values:
                demand = G.graph['trips'][key][value]
                # if there are positive trips
                if not ((demand == 0) or (demand == np.nan)):
                    path = paths[int(value)]
                    path_length = lengths[int(value)]
                    G.graph['sp'][(key,value)] = {'path': path, 'path_length': path_length}
                    path_edges = utils_graph.edges_from_path(path)
                    # refer to previous work or look for possible smarter way to update edges
                    for u,v in path_edges:
                        edge_id = G[u][v]['id']
                        y_aon[edge_id] += demand
        
        # Target solution y
        if (i > 1) and (method=='conjugate_frank_wolfe'):
            [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
            H = np.diag(list(map(edge_func_derivative, u,v,d,x)))
            alpha = _conjugate_step(x, y, y_aon, H)
            print('alpha = {}'.format(alpha))
            
            y = alpha*y + (1-alpha)*y_aon
        else:
            y = y_aon

        # store the data
        AEC  = _average_excess_cost(G)
        rel_gap = _relative_gap(G)
        print('i: {}'.format(i))
        print('AEC: {}'.format(AEC))
        print('rel gap: {}'.format(rel_gap))
        if collect_data and i > 1:
            data['AEC'].append(AEC)
            data['relative_gap'].append(rel_gap)
            data['x'].append(x)
            data['weight'].append(_edge_func(G,x,G.graph['edge_func']))
            data['objective'].append(objective(G,x))

        # convergence tests
        if AEC < aec_gap_tol  and i > 1:
            break

        if max_iter:
            if i > max_iter:
                break
        
        # generate new solution x based on previous solution x and new target y
        lam = 1
        if not i == 1:
            # Step 3: Update the current solution x <- lx* +(1-l)x for l in [0,1]
            if method == 'successive_averages':
                lam = 1/(i)
            else:
                lam = _line_search_fw(G, x, y, tol=line_search_tol)
            
        # convex combination of old solution and target solution
        x = lam*y + (1-lam)*x

        i+=1
        
    return G, data

def gradient_projection(G, 
    edge_func_derivative,
    aec_gap_tol=10**-4, 
    max_iter=None, 
    collect_data=False,
    alpha=1):
    # Update to store shortest paths that have been used 
    data = {'AEC':[], 'relative_gap':[], 'x':[], 'weight':[], 'objective':[]}
    no_edges = len(G.edges())
    x = np.zeros(no_edges,dtype="float64") 
    _update_edge_attribute(G, 'x', x)
    _update_edge_attribute(G, 'weight', _edge_func(G,x,G.graph['edge_func']))
    _update_edge_attribute(G, 'derivative', _edge_func(G,x,edge_func_derivative))
    G.graph['sp'] = defaultdict(lambda: None)

    # dictionary to store paths for an (origin, destination)
    # e.g. {(origin, destination):{(1,2,3): {'cost', 4, 'flow',1}} stores a path [1,2,3] with length 4 and flow 1
    G.graph['paths'] = defaultdict(lambda: [])
    G.graph['paths_1'] = defaultdict(lambda: {'D': [], 'h':[] })
    i = 0
    while True:
        print('Iteration i = {}--------------------\n'.format(i))
        
        x = np.zeros(no_edges,dtype="float64") 
        for key, values in G.graph['trips'].items():
            #print('computing shortest paths')
            lengths, all_paths = nx.single_source_dijkstra(G, source=key, target=None, weight='weight')
            #print('finished computing shortest paths for orgin {}'.format(key))
            #print(lengths)
            origin = key
            # iterate through the destinations for an origin
            for value in values:
                destination = value
                demand = G.graph['trips'][origin][destination]
                # if there are positive trips
                if not ((demand == 0) or (demand == np.nan)):
                    #print('computing shortest path for od pair = ({},{})'.format(origin, destination))
                    path = all_paths[int(destination)]
                    path_length = lengths[int(destination)]
                    #path_length, path = nx.single_source_dijkstra(G, source=key, target=int(value))
                    paths = G.graph['paths'][(origin, int(destination))]
                    
                    if not tuple(path) in paths:
                        
                        # G.graph['paths'][(origin, int(destination))][tuple(path)] = {
                        #     'flow':0
                        # }
                        edges_id = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(path)]
                        path_vector = np.zeros(no_edges,dtype="int32")
                        path_vector[edges_id] = 1
                        
                        G.graph['paths_1'][(origin, int(destination))]['D'].append(path_vector)
                        G.graph['paths_1'][(origin, int(destination))]['h'].append(0)
                        G.graph['paths'][(origin, int(destination))].append(tuple(path))
                    
                    G.graph['sp'][(origin,destination)] = {'path': path, 'path_length': path_length}
                        

                    if i == 0:
                        # assign all demand to single path
                        #print('assigning first path for od = ({},{})'.format(key, value))
                        G.graph['paths_1'][(origin, int(destination))]['h'][0] = demand
                        
        AEC  = _average_excess_cost(G)
        rel_gap  = _relative_gap(G)
        # print('rel gap: {}'.format(AEC))
        if i > 0:
            print('AEC: {}'.format(AEC))
            #time.sleep(2)

        # convergence tests
        if AEC < aec_gap_tol and i > 0:
            break

        if max_iter:
            if i > max_iter:
                break

        
        if collect_data and i > 1:
            data['AEC'].append(AEC)
            data['relative_gap'].append(rel_gap)
            data['x'].append(x)
            data['weight'].append(_edge_func(G,x,G.graph['edge_func']))
            data['objective'].append(objective(G,x))
        
        t = _get_np_array_from_edge_attribute(G, 'weight')
        t_prime = _get_np_array_from_edge_attribute(G, 'derivative')

        for key, values in G.graph['trips'].items():            
            origin = key
            # iterate through the destinations for an origin
            #print(origin)
            #print('shifting flow for od pair {}'.format(origin))
            for value in values:
                destination = value
                demand = G.graph['trips'][origin][destination]
                if not ((demand == 0) or (demand == np.nan)):
                    
                    # refer to previous work or look for possible smarter way to update edges
                    paths = G.graph['paths'][(origin, int(destination))]
                    
                    
                    
                    D = np.array(G.graph['paths_1'][(origin, int(destination))]['D']).T
                    h = np.array(G.graph['paths_1'][(origin, int(destination))]['h'])
                    c = np.dot(np.transpose(D),t)

                    if len(h) > 1:
                        
                        

                        # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
                        #start = time.time()
                        h_prime = shift_flow(G,D,c, h, t, t_prime,demand, alpha)
                        #end = time.time()
                        #print("Elapsed (with compilation) = %s" % (end - start))
                        
                        x += _edge_flow_from_paths(D,h_prime)
                        # # store the updated flow
                        G.graph['paths_1'][(origin, int(destination))]['h'] = h_prime.tolist()
                    
                    else:
                        x += _edge_flow_from_paths(D,h)

        
        _update_edge_attribute(G, 'x', x)
        _update_edge_attribute(G, 'weight', _edge_func(G,x, G.graph['edge_func']))
        _update_edge_attribute(G, 'derivative', _edge_func(G,x,edge_func_derivative))

        

        i+=1

    return G, data
    # Iteration
    # For every OD pair
    # Step 1: Generate an intial path set paths_rs
    # Step 2: find shortest path and add it to paths_rs if not already in paths_rs
    # Step 3: Shift flow among paths
        # hp = hp + mu*del_hp
        # del_hp = cp - avg_cp
        # cp - cost of path p between rs
        # avg_cp - average cost of paths between rs
        # mu in [0,min(hp/del_hp)]
        # to start with try min(hp/del_hp)/2 then move to line search
    # Step 4: Drop empty paths (or tag)
    # Step 5: if convergence satisfied, stop. Otherwise return to step 2

#@jit
def shift_flow(G,D,c, h, t, t_prime, demand, alpha):
    sp_id = np.argmin(c)

    g = c - c[sp_id]
    g[sp_id] = 0

    # get all non common links for each of the paths
    non_common_links = np.bitwise_xor(D, D[:,[sp_id]])

    H = np.dot(np.transpose(non_common_links), t_prime)
    
    # amend 
    H[sp_id] = 1
    v = -g/H
    #v = -g
    D_diff = D - D[:,[sp_id]]
    
    # update h, not that the update for the shortest path will be 0
    h_prime = np.maximum(np.zeros(len(h)), h+ alpha*v)
    h_prime[sp_id] += demand - np.sum(h_prime)

    return h_prime


# need to update docstrings and kwgs (clean up)
def conjugate_frank_wolfe(G, edge_func_derivative, **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, dat = link_based_method(G, 'conjugate_frank_wolfe', edge_func_derivative=edge_func_derivative, **kwargs)
    return G, data

# need to update docstrings and kwgs (clean up)
def frank_wolfe(G, **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, data = link_based_method(G, 'frank_wolfe', **kwargs)
    return G, data

def successive_averages(G,  **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, data = link_based_method(G, 'successive_averages',  **kwargs)
    return G, data

def total_system_travel_time(G):
    x = _get_np_array_from_edge_attribute(G, 'x')
    t = _get_np_array_from_edge_attribute(G, 'weight')
    return np.dot(x,t)

def _all_demand_on_fastest_paths(G):
    k = np.array([value['path_length'] for key, value in G.graph['sp'].items()])
    
    d = np.array([G.graph['trips'][key[0]][key[1]] for key, value in G.graph['sp'].items()])
    return np.dot(k,d)

def _relative_gap(G):
    return (total_system_travel_time(G)/_all_demand_on_fastest_paths(G))-1

def _average_excess_cost(G):
    d =  np.array([G.graph['trips'][key[0]][key[1]] for key, value in G.graph['sp'].items()])
    
    return (total_system_travel_time(G)-_all_demand_on_fastest_paths(G))/np.sum(d)
    
def _line_search_fw(G, x, y, tol=0.01):
    p = 0
    q = 1
    while True:
        alpha = (p+q)/2.0

        # dz/d_alpha derivative
        D_alpha = sum((y-x)*_edge_func(G, x + alpha*(y-x),G.graph['edge_func']))
        
        if D_alpha <= 0:
            p = alpha
        else:
            q = alpha
        if q-p < tol:
            break
        
    return (p+q)/2


def _line_search_path_based(G, D, h, del_h, mu_lim, tol=0.01):
    p = 0
    q = mu_lim
    
    del_x = _edge_flow_from_paths(D,del_h)
    while True:
        x = _get_np_array_from_edge_attribute(G, attr='x')
        alpha = (p+q)/2.0
        x += _edge_flow_from_paths(D,alpha*del_h)
        t = _edge_func(G,x)
        # dz/d_alpha derivative
        D_alpha = np.sum(t*del_x)
        print(D_alpha)
        print(alpha)
        
        if D_alpha <= 0:
            p = alpha
        else:
            q = alpha
        if q-p < tol:
            break
        
    return (p+q)/2

def _update_edge_attribute(G, attr, vector):
    d = dict(zip(sorted(G.edges()), vector))
    nx.set_edge_attributes(G, d, attr)
    return G

def _edge_func(G,x, func):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return list(map(func, u,v,d,x))

def _get_np_array_from_edge_attribute(G, attr):
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, attr).items())],dtype="float64")

def objective(G, x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return sum(list(map(G.graph['edge_func_integral'], u,v,d,x)))

def _conjugate_step(x, y, y_aon,H, epsilon= 0.01):
    denom = np.dot((y - x), np.dot(H, (y_aon - y)))
    if denom == 0:
        alpha = 0
    else:
        alpha = np.dot((y - x), np.dot(H, (y_aon - x)))/denom
        print('alpha before projection = {}'.format(alpha))
        if alpha > 1-epsilon:
            alpha = 1-epsilon
        if alpha < 0:
            alpha = 0
    return alpha

def _edge_path_matrix_od(G, origin, destination):
    paths = list(G.graph['paths'][(origin, destination)].keys())
    D = np.zeros((len(G.edges()),len(paths)))
    for indx, path in enumerate(paths):
        edges_id = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(path)]
        D[edges_id, indx] = 1

    return D

def _edge_path_matrix(G):
    edge_path_matrices = []
    for key, values in G.graph['trips'].items():
            # iterate through the destinations for an origin
            for value in values:
                edge_path_matrices.append(_edge_path_matrix_od(G, key, int(value)))

    D = edge_path_matrices.pop(0)
    for edge_path_matrix in edge_path_matrices:
        D = np.append(D, edge_path_matrix, axis=1)

    return D


def _path_flows(G, origin, destination):
    return np.array([dict['flow'] for key, dict in G.graph['paths'][(origin, destination)].items()],dtype="float64")

def _edge_flow_from_paths(D, h):
    x = np.dot(D,h)
    return x

# compute all path flow costs
def _path_flow_costs(G,origin=None, destination=None):
    x = _get_np_array_from_edge_attribute(G, attr='x')
    t = _edge_func(G,x,G.graph['edge_func'])
    if origin and destination:
        D_od = _edge_path_matrix_od(G, origin, destination)    
        c = np.dot(np.transpose(D_od),t)    
    else:
        D = _edge_path_matrix(G)
        c = np.dot(np.transpose(D),t)
    
    return c