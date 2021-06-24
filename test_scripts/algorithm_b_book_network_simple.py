import tapnx as tapnx
import networkx as nx
import tapnx.utils_graph as utils_graph
import tapnx.plot as plot_net
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

def initialise_bushes(G, trips):
    no_nodes = len(G.nodes())
    bushes = dict()
    for origin, destinations in trips.items():
        pred, dist = nx.dijkstra_predecessor_and_distance(G, origin, cutoff=None, weight="weight")
        # clean up pred to eliminate any case whereby there are more than 1 predeccesors
        # this is the case when there are multiple shortest paths
        pred = {key:value[0] for key, value in pred.items() if value}
        edges = get_edges_from_pred(pred)
        B = G.edge_subgraph(edges).copy()
        topological_order = list(nx.topological_sort(B))
        bushes[origin] = {'bush':B, 'x':np.zeros((no_nodes+1,no_nodes+1)), 'pred_L':pred, 'pred_U':{origin:None}, 'pot_pred_U':{origin:None}, 'dist_L':dist, 'dist_U':{origin:0}, 'pot_dist_U':{origin:0}, 'destinations':[], 'origin':origin, 'topological_order':topological_order}

    return bushes

def get_edges_from_pred(pred):
    edges = []
    # loop through all nodes in topological order.
        # store shortest path to current node as sp[node] = sp[pred] + dist(pred, node)
    for key, value in pred.items():
        u = value
        v = key
        edges.append((u,v))

    return edges

# assign initial flow to bush
def assign_initial_flows(bushes, trips):
    for origin, destinations in trips.items():
        bush_dict = bushes[origin]
        origin_trips = trips[origin]
        bush_dict = assign_initial_flow(bush_dict, origin_trips, destinations)

    return bushes

# assign initial flow to bush
def assign_initial_flow(bush_dict, origin_trips, destinations):
    pred_L = bush_dict['pred_L']
    origin = bush_dict['origin']
    x = bush_dict['x']
    for destination in destinations:
        demand = origin_trips[destination]
        
        if not ((demand == 0) or (demand == np.nan)):
            # for destination, work backwards throught pred tree, no need for min as initial bush will have
            # only one incoming link
            # note that the destination is a string (pandas to_dict)
            j = int(destination)
            i = int(destination)
            while not i == origin:
                i = pred_L[j]
                #print('adding {} for ({},{}) in bush {}'.format(demand,i,j,origin))
                x[i,j] += demand
                j = i

    return bush_dict

# relabels the min and max path costs to nodes
# returns the max cost differential for the convergence test.
def relabel(bush_dict, topological_order, c):
    # in topological order
    # Process vertices in topological order 
    dist_L = bush_dict['dist_L']    
    pred_L = bush_dict['pred_L']    
    dist_U = bush_dict['dist_U']    
    pred_U = bush_dict['pred_U'] 
    pot_dist_U = bush_dict['pot_dist_U']    
    pot_pred_U = bush_dict['pot_pred_U']  
    B = bush_dict['bush']
    del_c = 0
    origin = topological_order[0]
    #dist_L[origin] = 0
    #dist_U[origin] = 0
    for j in topological_order[1:]:
        (dist_L[j], dist_U[j], pot_dist_U[j], pred_L[j], pred_U[j], pot_pred_U[j]) = (np.inf, -np.inf, -np.inf, None, None, None)
        for (i,j) in B.in_edges([j]):
            # note that c_ij is computed using the total flow
            # test to see if the arc results in a better known min distance to j
            #print('checking arc ({},{})'.format(i,j))
            if dist_L[i] + c[i,j] < dist_L[j]:
                # update dist and pred
                #print('updating min distance for {}'.format(j))
                (dist_L[j] , pred_L[j]) = (dist_L[i] + c[i,j], i)

            # if there is flow on arc (i,j) then compute possible new max path cost and pred
            # otherwise only store dist, used when equilibriating a bush, we can filter nodes based
            # on them not having a max path pred. i.e. they only have a min max label with flow on
            if dist_U[i] + c[i,j] > dist_U[j]:
                
                if x[i,j] > 0:
                   (dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], i)
                #else:
                #    (dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], None)
                #(dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], i)

            if pot_dist_U[i] + c[i,j] > pot_dist_U[j]:
                (pot_dist_U[j], pot_pred_U[j]) = (pot_dist_U[i] + c[i,j], i)

            # compute the max cost differential between max and min paths to node
            if pred_U[j]:
                del_c = max(dist_U[j]-dist_L[j], del_c)

    return del_c

# drop any edge in the bush with zero flow
def drop_bush_edges(bush_dict):
    dropped = []
    candidate_edges_for_removal = []
    #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
    
    for u,v in bush_dict['bush'].edges():
        x = bush_dict['x']
        if x[u,v] == 0:
            #drop edge if it does not cause an isolate
            if bush_dict['bush'].in_degree(v) > 1:
                candidate_edges_for_removal.append((u,v))
    #print('\n\ncandidate_edges for removal: {}\n\n'.format(candidate_edges_for_removal))
    for u,v in candidate_edges_for_removal:
        # if bush_dict['bush'].in_degree(v) > 1:
        #     print('removing edge ({},{})'.format(u,v))
        #     bush_dict['bush'].remove_edge(u,v)
        #     dropped.append((u,v))
        # else:
        #     print('removing edge would result in a isolate({},{})'.format(u,v))
        #print('removing edge ({},{})'.format(u,v))
        bush_dict['bush'].remove_edge(u,v)
        dropped.append((u,v))
                
    
    #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
    #plt.show()
    return bush_dict, dropped

def improve_bush_dial(bush_dict, G, c):
    dist_L = bush_dict['dist_L']    
    pred_L = bush_dict['pred_L']    
    dist_U = bush_dict['dist_U']    
    pred_U = bush_dict['pred_U']  
    B = bush_dict['bush']
    # get topological order
    topological_order = bush_dict['topological_order']

    # to avoid processing nodes without a max flow path, compute max flow dist, but don't update pred
    B_edges = B.edges()
    # amend costs at the end, otherwise we can introduce a cycle
    edges_added= []
    for j in topological_order[1:]:
        for (i,j) in G.in_edges([j]):
            # this is patched, we should only consider adding if top order  of i is less than j
            if dist_L[i] + c[i,j] < dist_L[j] and not (i,j) in B_edges and not (j,i) in B_edges:
                edges_added.append((i,j))
                

    for (i,j) in edges_added:
        B.add_edge(i,j)
        (dist_L[j] , pred_L[j]) = (dist_L[i] + c[i,j], i)
        #print('adding edge ({},{})'.format(i,j))
          
    bush_dict['topological_order'] = list(nx.topological_sort(B))


def improve_bush_nie(bush_dict, G, c):
    dist_L = bush_dict['dist_L']    
    pred_L = bush_dict['pred_L']    
    dist_U = bush_dict['pot_dist_U']    
    pred_U = bush_dict['pot_pred_U']  
    B = bush_dict['bush']
    # get topological order
    topological_order = bush_dict['topological_order']

    edges_added = []
    # to avoid processing nodes without a max flow path, compute max flow dist, but don't update pred
    for j in topological_order[1:]:
        for (i,j) in G.in_edges([j]):
            if dist_U[i] + c[i,j] < dist_U[j]:
                edges_added.append((i,j))

    for (i,j) in edges_added:
        (dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], i)
        B.add_edge(i,j)
        #print('adding edge ({},{})'.format(i,j))
          
    bush_dict['topological_order'] = list(nx.topological_sort(B))

def get_branch_node(bush_dict, x, j, c, c_p):
    #print('getting branch node')
    pred_L = bush_dict['pred_L']
    pred_U = bush_dict['pot_pred_U']
    x = bush_dict['x']

    topological_order = bush_dict['topological_order']
    # pred_L, pred_u and x are local to the bush
    i_min = pred_L[j]
    j_min = j
    i_max = pred_U[j]
    j_max = j
    # as well as computing the branch node, we also compute the flow on the path, the cost and the 
    # derivative cost, these are used in shift flow
    x_min = x[i_min,j]
    x_max = x[i_max,j]
    c_min = c[i_min,j]
    c_max = c[i_max,j]
    c_p_min = c_p[i_min,j_min]
    c_p_max = c_p[i_max,j_max]
    topological_order_paths = [i_min, i_max, j]
    while not i_max == i_min:
        # while the current node on the min path is less than the max path (topologically) 
        while topological_order.index(i_min) < topological_order.index(i_max):
            #print('i_max = {}'.format(i_max))
            # track back on the min path
            # check that we are not at the origin (no predecessor)
            if i_max in pred_U:
                j_max = i_max
                i_max = pred_U[i_max]
                topological_order_paths.insert(0, i_max)
                
                x_max = min(x_max, x[i_max, j_max])
                c_max += c[i_max, j_max]
                c_p_max += c_p[i_max, j_max]
        # while the current node on the max path is less than the min path (topologically) 
        while topological_order.index(i_max) < topological_order.index(i_min):
            # track back on the min path
            # check that we are not at the origin (no predecessor)
            #print('i_min = {}'.format(i_min))
            if i_min in pred_L:
                j_min = i_min
                i_min = pred_L[i_min]
                topological_order_paths.insert(0, i_min)
                x_min = min(x_min, x[i_min, j_min])
                c_min += c[i_min, j_min]    
                c_p_min += c_p[i_min, j_min]

    topological_order_paths.pop(0)
    #print(topological_order_paths)
    return i_min, topological_order_paths, x_min, x_max, c_min, c_max, c_p_min, c_p_max

def update_path_flow(del_x, k, j, pred, x, c, c_p, coeff, x_b):
    # note that this is implemented as it is written in Dial 1999
    path_flow = np.inf
    path_cost = 0
    path_derivative = 0

    # set i = to the end node
    i = j
    while not i == k:
        # get arc indices
        j = i   
        i = pred[i]
        # update the flow of link u,v
        #print(i,j)
        x[i,j] += del_x
        
        x_b[i,j] += del_x
        if x_b[i,j] < 0:
            print('negative for k -> j = {}, {}'.format(k,j))
            time.sleep(5)
        # compute the path revised total flow
        #print(x[i,j])
        path_flow = min(path_flow, x_b[i,j])

        # compute revised arc cost and derivative
        c[i,j] = arc_cost(i, j, x[i,j], coeff)
        c_p[i,j] = arc_derivative(i, j, x[i,j], coeff)

        # compute the path cost and derivative
        path_cost += c[i,j]
        path_derivative += c_p[i,j]
    
    return path_flow, path_cost, path_derivative

def get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max):

    # if the cost of c_min is less that c_max, potential shift is x_max
    # It is possible that during equalising of the costs, x_min and x_max will effectively swap signs.
    # i.e. we may overshoot the optimal flow
    if c_min < c_max:
        del_x = x_max
    else:
        del_x = x_min
    # in the case that all flow has been removed from the max path, then del_x is 0, 
    # i.e. we should remove no flow and there is no path cost differential
    if del_x <= 0:
        return (0,0)
    # in the case where the sum of the cost derivatives are zero, shift all flow from max to min
    if c_p_min + c_p_max <= 0:
        # check that paths haven't swapped signs
        if c_min < c_max:
            del_x = x_max
        else:
            del_x = -x_min
    else:
        if c_min < c_max:
            # shift flow by the minimum available or the newton step
            del_x = min(del_x, (c_max-c_min)/(c_p_max + c_p_min))
        else:
            del_x = max(-del_x, (c_max-c_min)/(c_p_max + c_p_min))

    return del_x, np.abs(c_max-c_min)

# equilibriate paired alternative segments to a given node
def equilibriate_pas(bush_dict, j, x, c, c_p, coeff):
    pred_U = bush_dict['pred_U'] 
    pred_L = bush_dict['pred_L'] 
    x_b = bush_dict['x'] 
    
    (k, top_order_to_update, x_min, x_max, c_min, c_max, c_p_min, c_p_max) = get_branch_node(bush_dict, x_b, j, c, c_p)
    #print(x_min, x_max, c_min, c_max, c_p_min, c_p_max)
    #print('branch node = {}'.format(k))
    del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max)
    #print('c_min = {} and c_max = {}'.format(c_min, c_max))
    #print('x_min = {} and x_max = {}'.format(x_min, x_max))
    
    if x_max > 0:
        # while True:
        #     # shift flow
        #     print(del_c)
        #     if del_c < 0.000001:
        #         break
            
        #print('shifting {} flow for node {}'.format(del_x, j))
        x_min, c_min, c_p_min = update_path_flow(del_x, k, j, pred_L, x, c, c_p, coeff, x_b)
        x_max, c_max, c_p_max = update_path_flow(-del_x, k, j, pred_U, x, c, c_p, coeff, x_b)
        #print('c_min = {} and c_max = {}'.format(c_min, c_max))
        #print('x_min = {} and x_max = {}'.format(x_min, x_max))
        #del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max)

        if not x_max > 0:
            #print('x_max is 0')
            del_c = 0
            
            
                # relabel_pas 
            #del_c = relabel(bush_dict, top_order_to_update[1:], c)
            # for now relabel whole bush, update to pass only k->j of topological order
    else:
        del_c = 0

    #print('equilibriated pas for {}'.format(j))
            
    return top_order_to_update, del_c

def equilibriate_bush(bush_dict, x, c, c_p, coeff):
    topological_order = bush_dict['topological_order']
    pred_U = bush_dict['pred_U'] 
    pred_L = bush_dict['pred_L'] 
    #print(bush_dict['origin'])
    #print(topological_order)
    nodes_to_scan = topological_order[::-1][:-1]
    #print(nodes_to_scan)
    # scan nodes in reverse topological order, do not scan origin
    while True:
        max_del_c = 0
        for j in nodes_to_scan:
            # if there is more than one link into node j
            #print('pred_U[j] = {}'.format(pred_U[j]))
            #print('pred_L[j] = {}'.format(pred_L[j]))
            #if pred_U[j]:
            #print('\n\nscanning j={}\n\n'.format(j))
                #print('pred_L = {}'.format(pred_L[j]))
                #print('pred_U = {}'.format(pred_U[j]))
            top_order_to_update, del_c = equilibriate_pas(bush_dict, j, x, c, c_p, coeff)
                #del_c = max(del_c, relabel(bush_dict, top_order_to_update[1:], c))
            max_del_c = max(max_del_c, del_c)
            relabel(bush_dict, topological_order, c)
                #plot_bush(bush_dict)
                #plt.show()
        #print(max_del_c)
        #print('\n\n')
        #if max_del_c < 0.01:
            #time.sleep(1)
        break

def algorithm_main():
    return True
    # initialise bushes
    # assign initial flow
    # relabel
    # while True
        # for each origin
                # improve_bush
                # del_c = relabel
                # if del_c < epsilon
                    # break
                # equilibriate_bush for origin
                # drop edges

        # max_cost_diff = np.inf
        # for each origin
            # max_cost_diff = min(relabel , max_cost_diff)
        # if max_del_c < epsilon
            # break
    

# intialise bush 7
# assign initial flow
# relabel

# THEN

# improve_bush() and check acyclity...

def update_graph_flow(G, x):
    for u,v in G.edges():
        G[u][v]['x'] = np.round(x[u,v],2)

def update_graph_weight(G, c):
    for u,v in G.edges():
        G[u][v]['weight'] = np.round(c[u,v],2)
    
def compute_all_flow(bushes, no_nodes):
    x = np.zeros((no_nodes+1, no_nodes+1))
    for origin, bush_dict in bushes.items():
        x += bush_dict['x']

    return x

def compute_all_arc_costs(x,coeff):
    c = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            c[i,j] = arc_cost(i,j,x[i,j],coeff)
    return c

def compute_all_arc_derivatives(x,coeff):
    c_p = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            c_p[i,j] = arc_derivative(i,j,x[i,j],coeff)
    return c_p

def arc_cost(i,j,x,coeff):
    c_o = coeff['c_o']
    b = coeff['b']
    k = coeff['k']
    n = coeff['n']
    if k[i,j] > 0:
        return c_o[i,j]*(1+b[i,j]*(x/k[i,j])**n[i,j])
    else:
        return c_o[i,j]

def arc_derivative(i,j,x, coeff):
    c_o = coeff['c_o']
    b = coeff['b']
    k = coeff['k']
    n = coeff['n']
    if k[i,j] > 0:
        return (n[i,j]*c_o[i,j]*b[i,j]/(k[i,j]**n[i,j]))*x**(n[i,j]-1)
    else:
        return 0

def plot_bush(bush_dict, edge_label_attr='x'):
    B = bush_dict['bush']
    dist_L = bush_dict['dist_L']
    dist_U = bush_dict['dist_U']
    update_graph_flow(B, bush_dict['x'])
    update_graph_weight(B, c)
    #update_graph_flow(B, x)

    dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
    dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
    fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
    plot_net.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
    return fig, ax

def plot_bush_potential(bush_dict, edge_label_attr='x'):
    B = bush_dict['bush']
    dist_L = bush_dict['dist_L']
    pot_dist_U = bush_dict['pot_dist_U']
    update_graph_flow(B, bush_dict['x'])
    update_graph_weight(B, c)
    #update_graph_flow(B, x)

    dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
    dist_U_labels = {key: np.round(value, 2) for key, value in pot_dist_U.items()}
    fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
    plot_net.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
    plot_net.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
    return fig, ax

def plot_network(G, edge_label_attr='x'):
    fig, ax = tapnx.plot_graph(G, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
    return fig, ax

filename = 'bush_based_test_01'

G = tapnx.graph_from_csv(filename, nodes=True,trips=True, edge_attr=True)
a = utils_graph.get_np_array_from_edge_attribute(G,'a')
G = utils_graph.update_edge_attribute(G, 'weight',a)
G = utils_graph.update_edge_attribute(G, 'derivative_weight',a)
trips = G.graph['trips']
pos = G.graph['pos']

no_edges = G.number_of_edges()
no_nodes = len(G.nodes())

x = compute_all_flow(bushes, no_nodes)
c_o = np.zeros((no_nodes+1,no_nodes+1))
b = np.zeros((no_nodes+1,no_nodes+1))
k = np.zeros((no_nodes+1,no_nodes+1))
n = np.zeros((no_nodes+1,no_nodes+1))
for (u,v,d) in sorted(G.edges(data=True)):
    c_o[u,v] = d['a']
    b[u,v] = d['b']
    k[u,v] = d['c']
    n[u,v] = d['n']
coeff = {'c_o':c_o, 'b':b, 'k':k, 'n':n}
c = compute_all_arc_costs(x,coeff)
c_p = compute_all_arc_derivatives(x,coeff)

bushes = initialise_bushes(G, trips)
#bushes = assign_initial_flows(bushes, trips)


fig, ax = plot_network(G, edge_label_attr='x')
fig, ax = plot_bush(bushes[7], edge_label_attr='x')
plt.show()