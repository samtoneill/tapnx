import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# filename = 'MATLAB_test'

# df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
#     edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
#     trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
# )

# G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
# #G = tapnx.graph_positions_from_nodedf(G, df_nodes)
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
f,cf,x,cx,min_d = tapnx.gekko_optimise_column_gen(G,lam=1)
f,cf,x,cx,max_d = tapnx.gekko_optimise_column_gen(G,min_max_type=-1, lam=1)

f,cf,x,cx,min_tt = tapnx.gekko_optimise_column_gen(G,lam=0)
f,cf,x,cx,max_tt = tapnx.gekko_optimise_column_gen(G,min_max_type=-1, lam=0)

print(min_d)
print(max_d)
print(min_tt)
print(max_tt)