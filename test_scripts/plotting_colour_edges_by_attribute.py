import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import tapnx as tapnx
import networkx as nx

filename = 'siouxfalls'


G = tapnx.graph_from_TNTP(filename, nodes=True, edge_attr=True)
edge_color = tapnx.get_edge_colors_by_attr(G, 'a')

# plot network with edges coloured by edge attribute value
fig, ax = tapnx.plot_graph(
    G, edge_color=edge_color, node_size=200, node_labels=True,
    edge_labels=True, edge_label_attr='a'
)
fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)

# example plot of a path
fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
print(G.edges(data=True))
print(nx.dijkstra_path_length(G, 1, 20, weight=G.graph['weight']))
path = nx.dijkstra_path(G, 1, 20, weight=G.graph['weight'])
fig, ax = tapnx.plot_graph_path(G, path=path, ax=ax)


plt.show()