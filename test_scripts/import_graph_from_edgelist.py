import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import tapnx as tapnx
import networkx as nx


df_edges = pd.read_csv('../test_data/siouxfalls/SiouxFalls_net.csv')
df_nodes = pd.read_csv('../test_data/siouxfalls/SiouxFalls_node.csv', index_col=[0])
G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.graph_positions_from_nodedf(G, df_nodes)
G = tapnx.graph_edge_weight_func(
    G, weight_func = lambda u, v, d: d['a'] + d['b']*(d['x']/d['c'])**d['n']
)
edge_color = tapnx.get_edge_colors_by_attr(G, 'a')


fig, ax = tapnx.plot_graph(
    G, edge_color=edge_color, node_size=200, node_labels=True,
    edge_labels=True, edge_label_attr='a'
)
print(edge_color)
fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)


fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
print(G.edges(data=True))
G[1][2]['x']=100000
print(nx.dijkstra_path_length(G, 1, 20, weight=G.graph['weight']))
path = nx.dijkstra_path(G, 1, 20, weight=G.graph['weight'])
fig, ax = tapnx.plot_graph_path(G, path=path, ax=ax)

# test trips import 

df_trips = pd.read_csv('../test_data/chicagosketch/ChicagoSketch_trips.csv', index_col=0)
print(df_trips.head())

tapnx.graph_trips_from_tripsdf(G, df_trips)


plt.show()