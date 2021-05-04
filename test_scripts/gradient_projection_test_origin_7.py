import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'bush_based_test_06'

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)

#meta_data = tapnx.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
#df_edges = tapnx.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])

#df_nodes = tapnx.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
#df_trips = tapnx.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))



G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.trips_from_tripsdf(G, df_trips)
G.graph['name'] = filename
#G.graph['no_zones'] = int(meta_data['NUMBER OF ZONES'])
G.graph['no_nodes'] = len(G.nodes())
G.graph['first_thru_node'] = 1
G.graph['no_edges'] = len(G.edges())
G = tapnx.graph_positions_from_nodedf(G, df_nodes)


tol = 10**-3
max_iter = 300
#G, data_sa = tapnx.successive_averages(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
#plt.plot(data_sa['AEC'], label='Successive Averages')
#G, data_fw = tapnx.frank_wolfe(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
##plt.plot(data_fw['objective'])
# #print(data_fw['objective'])
#plt.plot(data_fw['AEC'], label='Frank Wolfe')
G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,alpha=0.5)
plt.plot(data['AEC'], label='Gradient Projection 1')
# #print(data_fw['x'][-1])
print(data['x'][-1])
#print(data['objective'][-1])
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()

tapnx.update_edge_attribute(G, 'x', np.round(data['x'][-1],1))

fig, ax = tapnx.plot_graph(G, edge_labels=True, edge_label_attr='x')
plt.show()