import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import tapnx as tapnx
import networkx as nx

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
# edge_color = tapnx.get_edge_colors_by_attr(G, 'a')


# fig, ax = tapnx.plot_graph(
#     G, edge_color=edge_color, node_size=200, node_labels=True,
#     edge_labels=True, edge_label_attr='a'
# )

# fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)


# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)

# #print(nx.dijkstra_path_length(G, 1, 3, weight=G.graph['weight']))
# path = nx.dijkstra_path(G, 1, 3, weight=G.graph['weight'])
# fig, ax = tapnx.plot_graph_path(G, path=path, ax=ax)


#plt.show()



tapnx.successive_averages(G, aec_gap_tol=2)
tapnx.frank_wolfe(G, aec_gap_tol=10**-4)
print(G.graph['sp'])
G, data = tapnx.successive_averages(G, aec_gap_tol=10**-2, collect_data=True)

plt.plot(data['AEC'], label='Successive Averages')

G, data = tapnx.frank_wolfe(G, aec_gap_tol=10**-2, collect_data=True)
plt.plot(data['AEC'], label='Frank Wolfe')

edge_func_derivative = lambda u, v, d, x: (d['b']*d['n']/(d['c']**d['n']))*x**(d['n']-1) 
G, data = tapnx.conjugate_frank_wolfe(
    G, 
    edge_func_derivative, 
    aec_gap_tol=10**-2,
    max_iter=5,
    collect_data=True) 
print(G.graph['sp'])
plt.plot(data['AEC'], label='Conjugate Frank Wolfe')

plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()
plt.show()