import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
from timeit import default_timer as timer
import time

filename = 'smallsiouxfalls'

G = tapnx.graph_from_csv(
    edges_filename = '../test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = '../test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = '../test_data/{}/{}_trips.csv'.format(filename, filename)
)

# Sioux Falls uses the BPR parameter of 0.15, thus we have a(1+b(x/c)^n) -> b = a +a*b
G = tapnx.graph_edge_weight_func(
    G, 
    edge_func = lambda u, v, d, x: d['a']*(1 + d['b']*(x/d['c'])**d['n']),
    edge_func_integral = lambda u, v, d, x: d['a']*x*(1 + (d['b']/(d['n']+1))*(x/d['c'])**d['n']) 
)
edge_color = tapnx.get_edge_colors_by_attr(G, 'a')


fig, ax = tapnx.plot_graph(
    G, edge_color=edge_color, node_size=200, node_labels=True,
    edge_labels=True, edge_label_attr='a'
)

fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)
plt.show()

# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)

# #print(nx.dijkstra_path_length(G, 1, 3, weight=G.graph['weight']))
# path = nx.dijkstra_path(G, 1, 3, weight=G.graph['weight'])
# fig, ax = tapnx.plot_graph_path(G, path=path, ax=ax)


#plt.show()

# length, path = nx.single_source_dijkstra(G, source=1, target=None)

# print(path)

tol = 10**-1

# G, data = tapnx.successive_averages(G, aec_gap_tol=tol, collect_data=True)
# plt.plot(data['AEC'], label='Successive Averages')



# G, data = tapnx.frank_wolfe(G, aec_gap_tol=tol, collect_data=True)
#plt.plot(data['AEC'], label='Frank Wolfe')

# objective = data['objective'][-1]
# NQ = []
# for u,v in G.edges():
#     H = G.copy()
#     H = tapnx.utils_graph.remove_edge(H,u,v)
    
#     H, data = tapnx.frank_wolfe(H, aec_gap_tol=tol, collect_data=True)
#     NQ.append(data['objective'][-1]/objective)
    
# plt.plot(NQ)
# plt.show()

#edge_func_derivative = lambda u, v, d, x: ((d['a']*d['b']*d['n'])/(d['c']**d['n']))*(x**(d['n']-1))

# x = np.ones(len(G.edges()),dtype="float64")
# [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
# print(d)
# print(list(map(edge_func_derivative, u,v,d,x)))


# G, data = tapnx.conjugate_frank_wolfe(
#     G, 
#     edge_func_derivative, 
#     aec_gap_tol=tol,
#     collect_data=True) 

# plt.plot(data['AEC'], label='Conjugate Frank Wolfe')

# plt.xlabel('No. Iterations')
# plt.ylabel('AEC')
# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(data['objective'])
# plt.show()

# plt.figure()
# plt.plot(data['relative_gap'])
# plt.show()
