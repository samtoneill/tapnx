import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'boyles190'

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)

G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.graph_positions_from_nodedf(G, df_nodes)
G = tapnx.graph_trips_from_tripsdf(G, df_trips)

G = tapnx.graph_edge_weight_func(
    G, 
    edge_func = lambda u, v, d, x: d['a'] + d['b']*(x/d['c'])**d['n'],
    edge_func_integral = lambda u, v, d, x: x*(d['a'] + (d['b']*(x/d['c'])**d['n'])/(d['n']+1))

)

edge_func_derivative = lambda u, v, d, x: (d['b']*d['n']/(d['c']**d['n']))*x**(d['n']-1) 

#G, data = tapnx.frank_wolfe(G, aec_gap_tol=10**-1, collect_data=True,max_iter=4,line_search_tol=10**-1.5)
#plt.plot(data['AEC'], label='Frank Wolfe')
#print(data['x'])
#print(data['weight'])
G, data = tapnx.gradient_projection(G, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=10**-2)
plt.plot(data['AEC'], label='Gradient Projection')
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()
plt.show()