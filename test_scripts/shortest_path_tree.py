import networkx as nx
import tapnx as tapnx
import tapnx.utils_graph as utils_graph
#from networkx.algorithms.shortest_paths.weighted import _weight_function
#from networkx.algorithms.shortest_paths.weighted import _dijkstra_multisource
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import time as time


def shortest_paths_pred(G,pred):
    paths = dict()
    dists = dict()
    # loop through all nodes in topological order.
        # store shortest path to current node as sp[node] = sp[pred] + dist(pred, node)
    for v in pred.keys():
        # access the predecessor
        if pred[v]:
            u = pred[v][0]
            dists[v] = dists[u] + G[u][v]['weight']
            paths[v] = paths[u] + [v]
        else:
            dists[v] = 0
            paths[v] = [v]

    return dists, paths

def get_edges_from_pred(pred):
    edges = []
    # loop through all nodes in topological order.
        # store shortest path to current node as sp[node] = sp[pred] + dist(pred, node)
    for v in pred.keys():
        # access the predecessor
        if pred[v]:
            for u in pred[v]:
                edges.append((u,v))

    return edges

def shortest_path_dag(G, s):
    adj_list = nx.to_dict_of_lists(G)
    stack = list(nx.topological_sort(G))
    dist = defaultdict(lambda: float("Inf"))
    paths = dict()
    dist[s] = 0
    paths[s] = [s]
    #print(stack)
    #print(dist)
    
    # Process vertices in topological order 
    for u in stack:
        # Update distances of all adjacent vertices 
        for v in adj_list[u]: 
            if dist[v] > dist[u] + G[u][v]['weight']: 
                dist[v] = dist[u] + G[u][v]['weight']
                paths[v] = paths[u] + [v]

    return dist, paths

def longest_path_dag(G, s):
    adj_list = nx.to_dict_of_lists(G)
    stack = list(nx.topological_sort(G))
    dist = defaultdict(lambda: -float("Inf"))
    paths = dict()
    dist[s] = 0
    paths[s] = [s]
    #print(stack)
    #print(dist)
    
    # Process vertices in topological order
    for u in stack:
        # Update distances of all adjacent vertices 
        for v in adj_list[u]: 
            if dist[v] < dist[u] + G[u][v]['weight']: 
                dist[v] = dist[u] + G[u][v]['weight']
                paths[v] = paths[u] + [v]

    return dist, paths

def divergence_node(shortest_path, longest_path):

    divergence_node = None
    # reverse the shortest list
    nodes = shortest_path[::-1]
    # scan the nodes in reverse order, exclude last node
    for u in shortest_path[::-1][1:]:
        if u in longest_path:
            divergence_node = u
            break
    
    return divergence_node

def algorithm_b(G, trips):

    no_edges = G.graph['no_edges']
    # compute shortest path trees for all origins
    # return a dict {origin:{B:networkx graph, x:numpy array (length is no edges in G not B!)}}
    bushes = initialise_bushes(G,trips,no_edges)
    #bushes = drop_edges(bushes)
    for origin, bush_dict in bushes.items():
            fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
            plt.show()
    x = np.zeros(no_edges)
    # edge attributes used in the calculation of edge travel times
    a = utils_graph.get_np_array_from_edge_attribute(G, 'a')
    b = utils_graph.get_np_array_from_edge_attribute(G, 'b')
    c = utils_graph.get_np_array_from_edge_attribute(G, 'c')
    n = utils_graph.get_np_array_from_edge_attribute(G, 'n')

    # update the travel times and derivatives
    t, t_prime = update_edge_labels(x,a,b,c,n)

    # update graph weights
    utils_graph.update_edge_attribute(G, 'weight', t)
    
    # update bush weights, required for shortest paths and longest paths
    bushes = update_bush_weights(bushes, t)
    # this computes U, L, shortest_paths, longest_paths
    bushes = update_labels(bushes)

    # all or nothing assignment
    bushes = assign_initial_flows(bushes, trips)

    # compute the total flow on edges from all bushes
    x = compute_total_flow(bushes, no_edges)
    
    # update the travel times and derivatives
    t, t_prime = update_edge_labels(x,a,b,c,n)

    # update bush weights, required for shortest paths and longest paths
    bushes = update_bush_weights(bushes, t)

    # this computes U, L, shortest_paths, longest_paths
    bushes = update_labels(bushes)
    
    # drop links (if shortest path only computed for destinations, not necessary)
    bushes = drop_edges(bushes)
    # this is needed as original bush connects all destinations, however after assignment, some edges 
    # have zero flow and are thus removed, this may disconnected nodes who have a demand of 0
    #bushes = clean_up_disconnected_nodes(bushes)
    # add shortcuts
    bushes = add_shortcut_edges(bushes, G, t)

    # update bush weights, required for shortest paths and longest paths
    bushes = update_bush_weights(bushes, t)
    
    i = 1
    while True:
        
        # this computes U, L, shortest_paths, longest_paths
        #bushes = update_labels(bushes)
        # using the shortest_paths and the current flow, compute AEC measure
        # requires the current flow and current shortest paths, i.e. calculate AEC after computing the new
        # shortest paths
        #AEC = aec()
        # convergence test
        #if AEC < tol:
        if i > 5:
            break

        for origin, bush_dict in bushes.items():
            print('equilibriating bush for origin {}'.format(origin))
            while True:
                # update all_bushes with new weights
                update_bush_weights(bushes, t)
                # update bush labels
                bush_dict = update_bush_labels(bush_dict)
                # equilibriate bush
                bush_dict = equilibriate_bush(bush_dict, t_prime, bushes, no_edges, a, b, c, n)
                # compute total flow
                x = compute_total_flow(bushes, no_edges)
                # compute t and t_prime
                t, t_prime = update_edge_labels(x,a,b,c,n)
                # update all_bushes with new weights
                update_bush_weights(bushes, t)
                #update bush labels
                bush_dict = update_bush_labels(bush_dict)
                # drop edges for bush
                bush_dict, dropped = drop_bush_edges(bush_dict)
                # add shortcuts for bush
                bush_dict, no_edges_added = add_shortcut_bush_edges(bush_dict, G, t, dropped)

                if no_edges_added == 0:
                    break

            
        # equilibriate the bushes by shifting flow from longest path to shortest path
        #bushes = equilibriate_bushes(bushes, t_prime)
        
        # compute the total flow on edges from all bushes
        #x = compute_total_flow(bushes, no_edges)

        # update the travel times and derivatives
        #t, t_prime = update_edge_labels(x,a,b,c,n)
        
        # update bush weights, required for shortest paths and longest paths
        #bushes = update_bush_weights(bushes, t)

        # this computes U, L, shortest_paths, longest_paths
        #bushes = update_labels(bushes)
        
        # drop links (if shortest path only computed for destinations, not necessary)
        #bushes = drop_edges(bushes)
        #bushes = clean_up_disconnected_nodes(bushes)
        # add shortcuts
        #bushes = add_shortcut_edges(bushes, G, t)
        # update bush weights, required for shortest paths and longest paths
        # MAY BE A MORE EFFICIENT WAY TO DO THIS
        #bushes = update_bush_weights(bushes, t)
        # for origin, bush_dict in bushes.items():
        #     fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
        #     plt.show()

        print(i)
        print(x)
        i += 1
        

# intialise a set of bushes for each origin 
# note that this computes a bush to every destination, update to only include destinations. subset of pred??
# returns a dict {origin: networkx Graph}
def initialise_bushes(G, trips, no_edges):
    bushes = dict()
    for origin, destinations in trips.items():
        pred, distance = nx.dijkstra_predecessor_and_distance(G, origin, cutoff=None, weight="weight")
        pred = {key:[value[0]] for key, value in pred.items() if value}
        edges = get_edges_from_pred(pred)
        B = G.edge_subgraph(edges).copy()
        bushes[origin] = {'bush':B, 'x':np.zeros(no_edges), 'L':{}, 'U':{}, 'L_paths':{}, 'U_paths':{}, 'destinations':[], 'origin':origin}

    return bushes

def assign_initial_flows(bushes, trips):
    for origin, destinations in trips.items():
        bush_dict = bushes[origin]
        for destination in destinations:
            demand = trips[origin][destination]
            
            if not ((demand == 0) or (demand == np.nan)):
                
                B = bush_dict['bush']
                demand = trips[origin][destination]
                shortest_path = bush_dict['L_paths'][int(destination)]
                path_edges = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]
                bush_dict['x'][path_edges] += demand
                bush_dict['destinations'].append(destination)

    return bushes

def compute_total_flow(bushes, no_edges):
    x = np.zeros(no_edges)
    for origin, B in bushes.items():
        x += B['x'] 
    return x

def update_edge_labels(x,a,b,c,n):
    t = _edge_func_np(x,a,b,c,n)
    t_prime = _edge_func_derivative_np(x,a,b,c,n)

    return t, t_prime

def update_bush_weights(bushes,t):
    for origin, bush_dict in bushes.items():
        B = bush_dict['bush']
        edge_ids = [B[u][v]['id'] for u,v in sorted(B.edges())]
        B = utils_graph.update_edge_attribute(B, 'weight', t[edge_ids])
    
    return bushes

# update the labels for each of the bushes
def update_labels(bushes):
    for origin, bush_dict in bushes.items():
        B = bush_dict['bush']
        bush_dict['L'], bush_dict['L_paths'] = shortest_path_dag(B,origin)
        bush_dict['U'], bush_dict['U_paths'] = longest_path_dag(B,origin)
    return bushes

def update_bush_labels(bush_dict):
    B = bush_dict['bush']
    origin = bush_dict['origin']
    bush_dict['L'], bush_dict['L_paths'] = shortest_path_dag(B,origin)
    bush_dict['U'], bush_dict['U_paths'] = longest_path_dag(B,origin)
    return bush_dict

# drop any edge in the bush with zero flow
def drop_bush_edges(bush_dict):
    dropped = []
    candidate_edges_for_removal = []
    #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
    
    for u,v in bush_dict['bush'].edges():
        edge_id = G[u][v]['id']
        x = bush_dict['x']
        if x[edge_id] == 0:
            #drop edge if it does not cause an isolate
            if bush_dict['bush'].in_degree(v) > 1:
                candidate_edges_for_removal.append((u,v))

    for u,v in candidate_edges_for_removal:
        if bush_dict['bush'].in_degree(v) > 1:
            #print('removing edge ({},{})'.format(u,v))
            bush_dict['bush'].remove_edge(u,v)
            dropped.append((u,v))
        #else:
        #    print('removing edge would result in a isolate({},{})'.format(u,v))
    

    #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
    #plt.show()
    return bush_dict, dropped

# drop any edge in the bush with zero flow
def drop_edges(bushes):
    for origin, bush_dict in bushes.items():
        candidate_edges_for_removal = []
        #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
        
        for u,v in bush_dict['bush'].edges():
            edge_id = G[u][v]['id']
            x = bush_dict['x']
            if x[edge_id] == 0:
                #drop edge if it does not cause an isolate
                if bush_dict['bush'].in_degree(v) > 1:
                    candidate_edges_for_removal.append((u,v))

        for u,v in candidate_edges_for_removal:
            if bush_dict['bush'].in_degree(v) > 1:
                #print('removing edge ({},{})'.format(u,v))
                bush_dict['bush'].remove_edge(u,v)
            #else:
            #    print('removing edge would result in a isolate({},{})'.format(u,v))
        

        #fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True)
        #plt.show()
    return bushes

def clean_up_disconnected_nodes(bushes):
    for origin, bush_dict in bushes.items():
        B = bush_dict['bush']
        isolates = list(nx.isolates(B))
        B.remove_nodes_from(isolates)

    return bushes

def add_shortcut_bush_edges(bush_dict, G, t, dropped):
    G_edges = set(G.edges())
    B = bush_dict['bush']
    origin = bush_dict['origin']
    U = bush_dict['U']
    B_edges = set(B.edges())
    diff_edges = list(G_edges.difference(B_edges))
    edges_for_addition = []
    topological_nodes = list(nx.topological_sort(B))
    #print('looking for shortcuts for bush with origin {}'.format(origin))
    for u,v in diff_edges:
        # check that node is reachable from origin, before adding edges
        if u in topological_nodes:
            #print('inspecting node {}'.format(u))
            if not (u,v) in dropped:
                edge_id = G[u][v]['id']
                if U[u] + t[edge_id] < U[v]:
                    print('adding edge ({},{}) as a shortcut'.format(u,v))
                    edges_for_addition.append((u,v))
            
    for u,v in edges_for_addition:
        B.add_edge(u,v)
        B[u][v]['id'] = G[u][v]['id']

    return bush_dict, len(edges_for_addition)

def add_shortcut_edges(bushes, G, t):
    G_edges = set(G.edges())
    for origin, bush_dict in bushes.items():
        B = bush_dict['bush']
        U = bush_dict['U']
        B_edges = set(B.edges())
        diff_edges = list(G_edges.difference(B_edges))
        edges_for_addition = []
        topological_nodes = list(nx.topological_sort(B))
        #print('looking for shortcuts for bush with origin {}'.format(origin))
        for u,v in diff_edges:
            # check that node is reachable from origin, before adding edges
            if u in topological_nodes:
                #print('inspecting node {}'.format(u))
                edge_id = G[u][v]['id']
                if U[u] + t[edge_id] < U[v]:
                    #print('adding edge ({},{}) as a shortcut'.format(u,v))
                    edges_for_addition.append((u,v))
            
        for u,v in edges_for_addition:
            B.add_edge(u,v)
            B[u][v]['id'] = G[u][v]['id']

    return bushes
    
    # get edges not currently in bush



    # for each bush
    # get topological ordering
    # calculate U and L labels and paths
    # for each destination in reverse topological order (i.e. last topological node first) in the bush
    #    get the longest p_u and shortest path p_l
    #    compute divergence node a and find alternative segments pi_u and pi_l
    #    perform newton shift for edges on pi_u and pi_l (make sure that shift is capped at min edge flow for pi_l)

def equilibriate_bush(bush_dict, t_prime, bushes, no_edges, a, b, c, n):
    B = bush_dict['bush']
    origin = bush_dict['origin']
    x = bush_dict['x']
    L = bush_dict['L']
    U = bush_dict['U']
    shortest_paths = bush_dict['L_paths']
    longest_paths = bush_dict['U_paths']
    topological_order = list(nx.topological_sort(B))
    
    

    print('\nequilibriate bush for origin {}\n'.format(origin))
    print(x)
    # scan nodes in reverse topological order 
    for u in topological_order[::-1]:
        #print('computing divergence and non common links for u={}'.format(u))
        shortest_path = shortest_paths[u]
        longest_path = longest_paths[u]
        #print('L={} and U={} labels for u={}'.format(L[u], U[u], u))
        if not (shortest_path == longest_path):
            #print('shortest path = {} and longest path ={}'.format(shortest_path, longest_path))
            div_node = divergence_node(shortest_path,longest_path)
            pi_l = shortest_path[shortest_path.index(div_node):]
            pi_u = longest_path[longest_path.index(div_node):]
            path_edges_l = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_l)]
            path_edges_u = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_u)]
            #print(utils_graph.edges_from_path(pi_l))
            #print(utils_graph.edges_from_path(pi_u))
            denominator = np.sum(t_prime[path_edges_l])+np.sum(t_prime[path_edges_u])
            numerator = (U[u]-U[div_node])-(L[u]-L[div_node])
            print('upper path = {} with travel time of {} - {} = {}'.format(pi_u, U[u], U[div_node], U[u]-U[div_node]))
            print('lower path = {} with travel time of {} - {} = {}'.format(pi_l, L[u], L[div_node], L[u]-L[div_node]))
            # print(pi_l)
            # print(numerator)
            # print(denominator)
            del_h = numerator/denominator
            min_flow_longest = np.min(x[path_edges_u])
            flow_to_shift = np.min([min_flow_longest, del_h])
            # print('diveregence node is={}'.format(a))
            # print('shortest path non common = {}'.format(pi_l))
            # print('longest path non common = {}'.format(pi_u))
            # print('del h = {}'.format(del_h))
            # print('minimum flow on longest non common = {}'.format(min_flow_longest))
            if flow_to_shift > 0:
                x[path_edges_l] += flow_to_shift
                x[path_edges_u] -= flow_to_shift 
                x_all = compute_total_flow(bushes, no_edges)
                print(x_all)
                # compute t and t_prime
                t, t_prime = update_edge_labels(x_all,a,b,c,n)
                print('t={}'.format(t))
                # update all_bushes with new weights
                #update_bush_weights(bushes, t)   
                #bush_dict = update_bush_labels(bush_dict)

    
    return bush_dict

def equilibriate_bushes(bushes, t_prime):
    for origin, bush_dict in bushes.items():
        B = bush_dict['bush']
        x = bush_dict['x']
        L = bush_dict['L']
        U = bush_dict['U']
        shortest_paths = bush_dict['L_paths']
        longest_paths = bush_dict['U_paths']
        topological_order = list(nx.topological_sort(B))
        
        

        print('\nequilibriate bush for origin {}\n'.format(origin))
        print(x)
        # scan nodes in reverse topological order 
        for u in topological_order[::-1]:
            print('computing divergence and non common links for u={}'.format(u))
            shortest_path = shortest_paths[u]
            longest_path = longest_paths[u]
            #print('L={} and U={} labels for u={}'.format(L[u], U[u], u))
            if not (shortest_path == longest_path):
                print('shortest path = {} and longest path ={}'.format(shortest_path, longest_path))
                a = divergence_node(shortest_path,longest_path)
                pi_l = shortest_path[shortest_path.index(a):]
                pi_u = longest_path[longest_path.index(a):]
                path_edges_l = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_l)]
                path_edges_u = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_u)]
                #print(utils_graph.edges_from_path(pi_l))
                #print(utils_graph.edges_from_path(pi_u))
                denominator = np.sum(t_prime[path_edges_l])+np.sum(t_prime[path_edges_u])
                numerator = (U[u]-U[a])-(L[u]-L[a])
                print('upper path = {} with travel time of {} - {} = {}'.format(pi_u, U[u], U[a], U[u]-U[a]))
                print('lower path = {} with travel time of {} - {} = {}'.format(pi_l, L[u], L[a], L[u]-L[a]))
                # print(pi_l)
                # print(numerator)
                # print(denominator)
                del_h = numerator/denominator
                min_flow_longest = np.min(x[path_edges_u])
                # print('diveregence node is={}'.format(a))
                # print('shortest path non common = {}'.format(pi_l))
                # print('longest path non common = {}'.format(pi_u))
                # print('del h = {}'.format(del_h))
                # print('minimum flow on longest non common = {}'.format(min_flow_longest))
        
                x[path_edges_l] += np.min([min_flow_longest, del_h])
                x[path_edges_u] -= np.min([min_flow_longest, del_h])

        # print(x)
        fig, ax = tapnx.plot_graph(bush_dict['bush'], edge_labels=True, edge_label_attr='weight')
        plt.show()
    return bushes


def _edge_func_np(x,a,b,c,n):
    return a*(1+b*(x/c)**n)

def _edge_func_derivative_np(x,a,b,c,n):
    return (a*b*n*x**(n-1))/(c**n)

filename = 'siouxfalls'

meta_data = tapnx.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
df_edges = tapnx.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])
df_nodes = tapnx.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
df_trips = tapnx.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))

filename = 'bush_based_test'

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

no_edges = G.graph['no_edges']

#source = 2
#target = None
#pred, distance = nx.dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight")

#print(pred)
#print(distance)

#lengths, all_paths = nx.single_source_dijkstra(G, source, target=None, weight='weight')
#print(lengths)
#print(all_paths)
#print(shortest_paths_pred(G,pred))

# create a subgraph involving the edges on the shortest paths
# check if a dag via networkx
# topological ordering (not necessary on first pass?)
# edges = get_edges_from_pred(pred)
# B = G.edge_subgraph(edges).copy()  


#dist_shortest, shortest_paths = shortest_path_dag(B,source)
#dist_longest, longest_paths = longest_path_dag(B,source)
#print(shortest_paths[21])
#print(dist_shortest[21])
#print(longest_paths[21])
#print(dist_longest[21])
#longest_path = [7,4,5,2]
#shortest_path = [7,4,1,2]
# a = divergence_node(shortest_path,longest_path)
# pi_l = shortest_path[shortest_path.index(a):]
# pi_u = longest_path[longest_path.index(a):]
# print(pi_l)
# print(pi_u)
# print(utils_graph.edges_from_path(pi_l))
# print(utils_graph.edges_from_path(pi_u))
#x = np.zeros(no_edges,dtype="float64") 
# path_edges = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_u)]
# x[path_edges] += 10
#print(nx.is_directed_acyclic_graph(G))
#print(nx.is_directed_acyclic_graph(B))
#print(list(nx.topological_sort(B)))
# fig, ax = tapnx.plot_graph(B)

# create a test bush


# B = nx.DiGraph()
# B.add_edges_from([(7,4), (4,1), (7,8), (4,5), (4,2), (1,2), (5,2), (8,9), (9,6),(2,3),(6,3)])
# for index, (u,v) in enumerate(sorted(B.edges(), key= lambda edge: (edge[0], edge[1]))):
#     B[u][v]['id'] = index

# bushes = {7:{'bush':B, 'x':np.array([1,7,1,1,5,5,3,7,3,3,3],dtype="float64"), 'L':{}, 'U':{}, 'L_paths':{}, 'U_paths':{}}}
# #x = np.array([1,7,1,1,5,5,3,7,3,3,3],dtype="float64")
# t = np.array([6,51,6,42,27,27,11,51,11,11,11],dtype="float64")
# t_prime = np.array([4,14,4,4,10,10,6,14,6,6,6],dtype="float64")

# update_bush_weights(bushes, t)
# update_labels(bushes)
# equilibriate_bushes(bushes, t_prime)
# L = bushes[7]['L']
# U = bushes[7]['U']
# shortest_paths = bushes[7]['L_paths']
# longest_paths = bushes[7]['U_paths']
#L, shortest_paths = shortest_path_dag(B,7)
#U, longest_paths = longest_path_dag(B,7)
#print(dist_shortest)
#print(dist_longest)
#print(shortest_paths)
#print(longest_paths)
# topological_order = list(nx.topological_sort(B))
# for u in topological_order[::-1]:
#     print('computing divergence and non common links for u={}'.format(u))
#     shortest_path = shortest_paths[u]
#     longest_path = longest_paths[u]
#     if not (shortest_path == longest_path):
#         a = divergence_node(shortest_path,longest_path)
#         pi_l = shortest_path[shortest_path.index(a):]
#         pi_u = longest_path[longest_path.index(a):]
#         path_edges_l = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_l)]
#         path_edges_u = [B[u][v]['id'] for u,v in utils_graph.edges_from_path(pi_u)]
#         print(utils_graph.edges_from_path(pi_l))
#         print(utils_graph.edges_from_path(pi_u))
#         denominator = np.sum(t_prime[path_edges_l]+t_prime[path_edges_u])
#         numerator = (U[u]-U[a])-(L[u]-L[a])
#         del_h = numerator/denominator
#         min_flow_longest = np.min(x[path_edges_u])
#         print('diveregence node is={}'.format(a))
#         print('shortest path non common = {}'.format(pi_l))
#         print('longest path non common = {}'.format(pi_u))
#         print('del h = {}'.format(del_h))
#         print('minimum flow on longest non common = {}'.format(min_flow_longest))
#         print(x)
#         x[path_edges_l] += np.min([min_flow_longest, del_h])
#         x[path_edges_u] -= np.min([min_flow_longest, del_h])
#         print(x)

# fig, ax = tapnx.plot_graph(G, edge_labels=True, edge_label_attr='weight')
# plt.show()
print(trips)
algorithm_b(G, trips)
