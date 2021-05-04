import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'amendedboyles190'

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)

G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.graph_positions_from_nodedf(G, df_nodes)
G = tapnx.graph_trips_from_tripsdf(G, df_trips)

# Sioux Falls uses the BPR parameter of 0.15, thus we have a(1+b(x/c)^n) -> b = a +a*b
G = tapnx.graph_edge_weight_func(
    G, 
    edge_func = lambda u, v, d, x: d['a']*(1 + d['b']*(x/d['c'])**d['n']),
    edge_func_integral = lambda u, v, d, x: d['a']*x*(1 + (d['b']/(d['n']+1))*(x/d['c'])**d['n']) 
)

edge_func_derivative = lambda u, v, d, x: ((d['a']*d['b']*d['n'])/(d['c']**d['n']))*(x**(d['n']-1))

tol = 10**-3

#G, data = tapnx.successive_averages(G, aec_gap_tol=tol, collect_data=True)

#G, data = tapnx.frank_wolfe(G, aec_gap_tol=10**-1, collect_data=True)
#plt.plot(data['AEC'], label='Frank Wolfe')
G, data = tapnx.gradient_projection(G, edge_func_derivative=edge_func_derivative,aec_gap_tol=10**-3,max_iter=10,collect_data=True)
#plt.plot(data['AEC'], label='Gradient Projection')
# plt.xlabel('No. Iterations')
# plt.ylabel('AEC')
# plt.yscale('log')
# plt.legend()
plt.show()