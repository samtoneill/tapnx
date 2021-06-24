import tapnx as tapnx
import tapnx.utils_graph as utils_graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time as time


# intialise a set of bushes for each origin 
# note that this computes a bush to every destination
# returns a dict {origin: networkx Graph}
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
        bushes[origin] = {'bush':B, 'x':np.zeros((no_nodes+1,no_nodes+1)), 'pred_L':pred, 'pred_U':{}, 'dist_L':dist, 'dist_U':{}, 'destinations':[], 'origin':origin, 'topological_order':topological_order}

    return bushes


def assign_initial_flows(bushes, trips):
    for origin, destinations in trips.items():
        bush_dict = bushes[origin]
        pred_L = bush_dict['pred_L']
        x = bush_dict['x']
        
        for destination in destinations:
            demand = trips[origin][destination]
            
            if not ((demand == 0) or (demand == np.nan)):
                # for destination, work backwards throught pred tree, no need for min as initial bush will have
                # only one incoming link
                # note that the destination is a string (pandas to_dict)
                j = int(destination)
                i = int(destination)
                while not i == origin:
                    i = pred_L[j]
                    print('adding {} for ({},{}) in bush {}'.format(demand,i,j,origin))
                    x[i,j] += demand
                    j = i

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
    print('\n\ncandidate_edges for removal: {}\n\n'.format(candidate_edges_for_removal))
    for u,v in candidate_edges_for_removal:
        if bush_dict['bush'].in_degree(v) > 1:
            print('removing edge ({},{})'.format(u,v))
            bush_dict['bush'].remove_edge(u,v)
            dropped.append((u,v))
        else:
            print('removing edge would result in a isolate({},{})'.format(u,v))
    

    #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
    #plt.show()
    return bush_dict, dropped

def improve_bush_nie(bush_dict, c, G):
    topological_order = bush_dict['topological_order']
    B = bush_dict['bush']
    dist_L = bush_dict['dist_L']
    pred_L = bush_dict['pred_L']
    # do not consider the origin
    for i in topological_order[1:]:
        print('inspecting node {} for improvements'.format(i))
        for (i,j) in G.out_edges(i):
            #if dist_U[j] == -np.inf:
            # we can use this trick to circumnavigate -np.inf
            # note that the min label must equal the max label as there will only be one link into an edge
            # 
            if dist_L[i] + c[i,j] < dist_L[j]:
                print('adding edge ({},{})'.format(i,j))
                B.add_edge(i,j)
                #(dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], i)
                #pred_U[j] = i

def update_trees(topological_order, bush_dict, x, c):
    """
    update the labels for the bush between k and n.
    examine all incoming arcs to the set {k+1, k+2, ..., n-1}
    FOR MIN PATH
    if the incoming arc (i,j) provides a better min_path then update min_path and pred for the node j
    FOR MAX PATH
    if the incoming arc (i,j) has flow and j is a predecessor and max_path_i + cij > max_path_j
        update labels
    
    Parameters
    ----------
    k : int
        exclusive lower bound on nodes to label
    n : int
        exclusive upper bound on nodes to label
    bush_dict: dict
        contains the bush and information about the bush 
    x: numpy 2d array
        flows of arcs (i,j) - this is computed with all the flow from all bushes
    c: numpy 2d array
        costs of arcs (i,j) - this is computed with all the flow from all bushes
    Returns
    -------
    del_c : float
        maximum difference of: max path cost - min path cost
    """

    dist_L = bush_dict['dist_L']    
    dist_U = bush_dict['dist_U']    
    pred_L = bush_dict['pred_L']    
    pred_U = bush_dict['pred_U']
    B = bush_dict['bush']
    x = bush_dict['x']

    del_c = 0

    for j in topological_order:
        #print('in update_trees, j = {}'.format(j))
        # check to see if there is no incoming min path, i.e. the origin
        if not (j in pred_L):
            dist_U[j] = 0
        else:    
            (dist_L[j], dist_U[j], pred_L[j], pred_U[j]) = (np.inf, -np.inf, None, None)

            for (i,j) in B.in_edges([j]):
                # note that c_ij is computed using the total flow
                # test to see if the arc results in a better known min distance to j
                #print('checking arc ({},{})'.format(i,j))
                if dist_L[i] + c[i,j] < dist_L[j]:
                    # update dist and pred
                    #print('updating min distance for {}'.format(j))
                    (dist_L[j] , pred_L[j]) = (dist_L[i] + c[i,j], i)
                
                # if a max path exists to predecessor, there is flow on (i,j) and arc results in better known max dist
                if (i in dist_U) and (x[i,j] > 0) and (dist_U[i] + c[i,j] > dist_U[j]):
                #if (i in dist_U) and (dist_U[i] + c[i,j] > dist_U[j]):
                #if dist_U[i] + c[i,j] > dist_U[j]:
                    # update dist and pred
                    (dist_U[j], pred_U[j]) = (dist_U[i] + c[i,j], i)
                
                # compute the best known maximum distance between min and max paths
                # only compute if there is a max path, i.e. a pred for j!!
                if pred_U[j]:
                    del_c = np.max([dist_U[j]-dist_L[j], del_c])

    return del_c

def improve_bush(bush_dict, c, G):
    
    # minimum distance labels for bush
    dist_L = bush_dict['dist_L']

    # minimum distance predecessors for bush
    pred_L = bush_dict['pred_L']

    # get topological order
    topological_order = bush_dict['topological_order']

    B = bush_dict['bush']
    x_b = bush_dict['x']
    
    # scan in reverse topological order to the origin
    for indx, i in enumerate(topological_order):
        #print('inspecting node {} for improvements'.format(i))
        for (i,j) in G.out_edges(i):
            #print('j = {}'.format(j))
            # proposition 3 Nie, cannot add a link (i,j) not in topological order
            #print(topological_order[indx:])
            #if j in topological_order[indx:]:
            
            # NEED TO UPDATE TO BE BUSH x
            # necessary as we could include a chain of links that update the dist..
            if not x_b[j,i] > 0:
                if dist_L[i] + c[i,j] < dist_L[j]:
                    # update dist and pred
                    #print('updating min distance for {}'.format(j))
                    (dist_L[j] , pred_L[j]) = (dist_L[i] + c[i,j], i)
                    B.add_edge(i,j)
                    if (j,i) in B.edges():
                        B.remove_edge(j,i)

    # finally ammend topological order
    bush_dict['topological_order'] = list(nx.topological_sort(B))

def get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max):
    """
    get flow and cost differentials
    
    Parameters
    ----------
    x_min : float
        total flow on minimum path
    x_max : float
        total flow on maximum path
    c_min : float 
        cost of minimum path
    c_max : float 
        cost of maximum path
    c_prime_min : float 
        derivative cost of minimum path
    c_prime_max : float 
        derivative cost of maximum path
    Returns
    -------
    del_x : float
        flow differential
    del_c : float 
        cost differential used for checking paths have equalised
    
    """
    print('c_min = {} and c_max = {}'.format(c_min, c_max))
    print('x_min = {} and x_max = {}'.format(x_min, x_max))
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
    print(del_x)
    print((c_max-c_min)/(c_p_max + c_p_min))
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

def get_branch_node(j, pred_L, pred_U, x, c, c_p, topological_order):
    print('getting branch node')
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
            if i_min in pred_L:
                j_min = i_min
                i_min = pred_L[i_min]
                topological_order_paths.insert(0, i_min)
                x_min = min(x_min, x[i_min, j_min])
                c_min += c[i_min, j_min]    
                c_p_min += c_p[i_min, j_min]

    topological_order_paths.pop(0)
    
    return i_min, topological_order_paths, x_min, x_max, c_min, c_max, c_p_min, c_p_max

def update_path_flow(del_x, k, j, pred, x, c, c_p, x_b):

    """
    update path flows for paired alternative segments
    
    Parameters
    ----------
    del_x : float
        flow to shift
    k : int
        start node of path to j
    j : int
        end node of path
    pred : dict    
        node predecessor-arc array
    Returns
    -------
    path_flow : float
        path p's revised total flow. i.e. the minimum amount of flow for the path that could be moved
            to another path k->j to ensure flow conservation
    path_cost : float
        path p's revised cost
    path_derivative : float
        path p's revised cost derivative 
    """
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
        
        x[i,j] += del_x
        print('updating x_b from {} to {}'.format(x_b[i,j], x_b[i,j] + del_x))
        x_b[i,j] += del_x
        # compute the path revised total flow
        path_flow = min(path_flow, x[i,j])

        # compute revised arc cost and derivative
        c[i,j] = arc_cost(i, j, x[i,j])
        c_p[i,j] = arc_derivative(i, j, x[i,j])

        # compute the path cost and derivative
        path_cost += c[i,j]
        path_derivative += c_p[i,j]

    return path_flow, path_cost, path_derivative

def equalise_path_cost(k, j, pred_L, pred_U, x_min, x_max, c_min, c_max, c_p_min, c_p_max, epsilon, x, c, c_p, x_b):
    """
    equalised path 
    
    Parameters
    ----------
    k : int
        start node of path to j
    j : int
        end node of path
    x_min : float
        total flow on minimum path
    x_max : float
        total flow on maximum path
    c_min : float 
        cost of minimum path
    c_max : float 
        cost of maximum path
    c_prime_min : float 
        derivative cost of minimum path
    c_prime_max : float 
        derivative cost of maximum path
    Returns
    -------
    
    """
    print('equalising path flow between {} and {}'.format(k,j))
    # get the flow and cost differentials
    del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max)
    # while the cost differential of a path segment is sufficiently large, shift flow
    while epsilon < del_c:
        #print('del_c = {}'.format(del_c))
        x_min, c_min, c_p_min = update_path_flow(del_x, k, j, pred_L, x, c, c_p, x_b)
        x_max, c_max, c_p_max = update_path_flow(-del_x, k, j, pred_U, x, c, c_p, x_b)
        #print(del_c)
        # get the update flow and cost differentials
        del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_p_min, c_p_max)
        
    print('equalised to {}'.format(del_c))
    # return control 

def shift_flow(bush_dict, x, c, c_p, epsilon):
    print('shifting flow for bush {}'.format(bush_dict['origin']))
    topological_order = bush_dict['topological_order']
    dist_L = bush_dict['dist_L']    
    dist_U = bush_dict['dist_U']    
    pred_L = bush_dict['pred_L']    
    pred_U = bush_dict['pred_U']
    x_b = bush_dict['x']

    k_hat = None
    # scan in reverse topological order
    for j in topological_order[::-1]:
        if j in pred_U and dist_L[j] < dist_U[j]:
            branch_node, top_order_to_update, x_min, x_max, c_min, c_max, c_p_min, c_p_max = get_branch_node(j, pred_L, pred_U, x, c, c_p, topological_order)
            #print('branch node = {} for node {}'.format(branch_node,j))
            equalise_path_cost(branch_node, j, pred_L, pred_U, x_min, x_max, c_min, c_max, c_p_min, c_p_max, epsilon, x, c, c_p, x_b)
            # don't include the branch node for update
            # IS THIS WRONG??
            update_trees(top_order_to_update[1:], bush_dict, x, c)
            #update_trees(topological_order, bush_dict, x, c)
            k_hat = branch_node
    return k_hat

def equilibriate_bush(bush_dict, x, c, epsilon=0.01):
    print('equilibrating bush {}'.format(bush_dict['origin']))
    topological_order = bush_dict['topological_order']
    # update tree
    while True:
        k_hat = shift_flow(bush_dict, x, c, c_p,epsilon)
        k_hat_index = topological_order.index(k_hat)
        del_c = update_trees(topological_order[k_hat_index:], bush_dict, x, c)
        
        #del_c = update_trees(topological_order, bush_dict, x, c)
        #print(del_c)
        if del_c < epsilon:
            break
    return True

def compute_all_flow(bushes, no_nodes):
    x = np.zeros((no_nodes+1, no_nodes+1))
    for orgin, bush_dict in bushes.items():
        x += bush_dict['x']

    return x

def compute_all_arc_costs(x):
    c = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            c[i,j] = arc_cost(i,j,x[i,j])
    return c

def compute_all_arc_derivatives(x):
    c_p = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            c_p[i,j] = arc_derivative(i,j,x[i,j])
    return c_p

def update_graph_weight(G, c):
    for u,v in G.edges():
        G[u][v]['weight'] = np.round(c[u,v],2)

def update_graph_flow(G, x):
    for u,v in G.edges():
        print(x[u,v])
        G[u][v]['x'] = np.round(x[u,v],2)

def arc_cost(i,j,x):
    if k[i,j] > 0:
        return c_o[i,j]*(1+b[i,j]*(x/k[i,j])**n[i,j])
    else:
        return 0

def arc_derivative(i,j,x):
    if k[i,j] > 0:
        return n[i,j]*c_o[i,j]*b[i,j]/(k[i,j]**n[i,j])*x**(n[i,j]-1)
    else:
        return 0

filename = 'bush_based_test_06'

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)

G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.graph_positions_from_nodedf(G,df_nodes)
G = tapnx.trips_from_tripsdf(G, df_trips)
trips = G.graph['trips']
a = utils_graph.get_np_array_from_edge_attribute(G,'a')
G = utils_graph.update_edge_attribute(G, 'weight',a)
G = utils_graph.update_edge_attribute(G, 'derivative_weight',a)


pos = G.graph['pos']

no_edges = G.graph['no_edges']
no_nodes = len(G.nodes())
bushes = initialise_bushes(G, trips)
bushes = assign_initial_flows(bushes, trips)
x = compute_all_flow(bushes, no_nodes)
c_o = np.zeros((no_nodes+1,no_nodes+1))
b = np.zeros((no_nodes+1,no_nodes+1))
k = np.zeros((no_nodes+1,no_nodes+1))
n = np.zeros((no_nodes+1,no_nodes+1))
for (u,v,d) in G.edges(data=True):
    c_o[u,v] = d['a']
    b[u,v] = d['b']
    k[u,v] = d['c']
    n[u,v] = d['n']

c = compute_all_arc_costs(x)
c_p = compute_all_arc_derivatives(x)

#update_trees(bushes[7]['topological_order'], bushes[7], x, c)

#update_graph_weight(B,c)
#update_graph_flow(B,x)

edge_label_attr = 'x'


# B = bushes[7]['bush']

# dist_L = bushes[7]['dist_L']    
# dist_U = bushes[7]['dist_U']    
# pred_L = bushes[7]['pred_L']    
# pred_U = bushes[7]['pred_U']

# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')

epsilon = 0.01


#improve_bush(bushes[7], c, G)
# drop_bush_edges(bushes[7])

# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
# plt.show()
# improve_bush_nie(bushes[7], c, G)
# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
# plt.show()
# del_c = update_trees(bushes[7]['topological_order'], bushes[7], x, c)
# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
# plt.show()
# print('attempt to equilibriate bush')
# equilibriate_bush(bushes[7], x, c, epsilon=epsilon)

# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
# plt.show()

bush_no = 7


for i in range(1):
    for bush_no in [7,9,17,19]:
        B = bushes[bush_no]['bush']

        dist_L = bushes[bush_no]['dist_L']    
        dist_U = bushes[bush_no]['dist_U']    
        pred_L = bushes[bush_no]['pred_L']    
        pred_U = bushes[bush_no]['pred_U']
        update_trees(bushes[bush_no]['topological_order'], bushes[bush_no], x, c)
        drop_bush_edges(bushes[bush_no])
        improve_bush_nie(bushes[bush_no], c, G)
        update_trees(bushes[bush_no]['topological_order'], bushes[bush_no], x, c)
        update_graph_weight(B,c)
        update_graph_flow(B,bushes[bush_no]['x'])
        dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
        dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
        fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
        utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
        utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')

        equilibriate_bush(bushes[bush_no], x, c, epsilon=epsilon)
                
        update_graph_weight(B,c)
        update_graph_flow(B,bushes[bush_no]['x'])
        dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
        dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
        fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
        utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
        utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
        print(bushes[bush_no]['x'])
        plt.show()
        
    


# B = bushes[bush_no]['bush']

# dist_L = bushes[bush_no]['dist_L']    
# dist_U = bushes[bush_no]['dist_U']    
# pred_L = bushes[bush_no]['pred_L']    
# pred_U = bushes[bush_no]['pred_U']
# update_graph_weight(B,c)
# update_graph_flow(B,bushes[bush_no]['x'])

# dist_L_labels = {key: np.round(value, 2) for key, value in dist_L.items()}
# dist_U_labels = {key: np.round(value, 2) for key, value in dist_U.items()}
# fig, ax = tapnx.plot_graph(B, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)
# utils_graph.draw_additional_labels(B, dist_L_labels, pos, -0.6, ax, font_color='g')
# utils_graph.draw_additional_labels(B, dist_U_labels, pos,  0.6, ax, font_color='r')
# plt.show()


update_graph_weight(G,c)
update_graph_flow(G,x)

fig, ax = tapnx.plot_graph(G, edge_labels=True, edge_label_attr=edge_label_attr, node_size=200)


# IT IS LIKELY THAT A BUSH CANNOT ADD AN ARC (i,j) AS FLOW EXISTS ON (j,i) from another bush.
# AMEND TO COMPUTE BUSHES VIA xb, which will allow this...

#print(x)

plt.show()