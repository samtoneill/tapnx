""" Network Equilibrium algorithms for TAP """

from collections import defaultdict
import numpy as np
import networkx as nx

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
        _update_edge_attribute(G, 'weight', _edge_func(G,x))
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
        #print('AEC: {}'.format(AEC))
        #print('rel gap: {}'.format(rel_gap))
        if collect_data and i > 1:
            data['AEC'].append(AEC)
            data['relative_gap'].append(rel_gap)
            data['x'].append(x)
            data['weight'].append(_edge_func(G,x))
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
                lam = _line_search(G, x, y, tol=line_search_tol)
            
        # convex combination of old solution and target solution
        x = lam*y + (1-lam)*x

        i+=1
        
    return G, data

# need to update docstrings and kwgs (clean up)
def conjugate_frank_wolfe(G, edge_func_derivative, **lbm_kwargs):
    kwargs = {k: v for k, v in lbm_kwargs.items()}
    G, data = link_based_method(G, 'conjugate_frank_wolfe', edge_func_derivative=edge_func_derivative, **kwargs)
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
    d = d = np.array([G.graph['trips'][key[0]][key[1]] for key, value in G.graph['sp'].items()])
    return (total_system_travel_time(G)-_all_demand_on_fastest_paths(G))/np.sum(d)
    
def _line_search(G, x, y, tol=0.01):
    p = 0
    q = 1
    while True:
        alpha = (p+q)/2.0

        # dz/d_alpha derivative
        D_alpha = sum((y-x)*_edge_func(G, x + alpha*(y-x)))
        
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

def _edge_func(G,x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return list(map(G.graph['edge_func'], u,v,d,x))

def _get_np_array_from_edge_attribute(G, attr):
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, attr).items())])

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