import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# filename = 'amendedboyles190'

# df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
#     edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
#     nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
#     trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
# )

# G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
# G = tapnx.graph_positions_from_nodedf(G, df_nodes)
# G = tapnx.trips_from_tripsdf(G, df_trips)

# G.graph['first_thru_node'] = 0

filename = 'siouxfalls'

meta_data = tapnx.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
df_edges = tapnx.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])

#df_nodes = tapnx.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
df_trips = tapnx.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))


G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.trips_from_tripsdf(G, df_trips)
G.graph['name'] = filename
G.graph['no_zones'] = int(meta_data['NUMBER OF ZONES'])
G.graph['no_nodes'] = int(meta_data['NUMBER OF NODES'])
G.graph['first_thru_node'] = int(meta_data['FIRST THRU NODE'])
G.graph['no_edges'] = int(meta_data['NUMBER OF LINKS'])


# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tapnx.milp_tap(G)