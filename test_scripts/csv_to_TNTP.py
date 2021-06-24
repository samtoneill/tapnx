import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np

filename = 'bush_based_test_06'


edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename)
trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
df_edges = pd.read_csv(edges_filename, index_col=[0])
df_trips = pd.read_csv(trips_filename, index_col=0)
G = tapnx.graph_from_csv(filename, nodes=False, trips=True, edge_attr=True)
no_nodes = G.number_of_nodes()
no_edges = G.number_of_edges()
edges_filename_output = 'test_data/{}/{}_net.txt'.format(filename, filename)
trips_filename_output = 'test_data/{}/{}_trips.txt'.format(filename, filename)


tapnx.pandas_net_to_TNTP(df_edges, edges_filename_output, no_nodes, no_edges)
tapnx.pandas_trips_to_TNTP(df_trips, trips_filename_output, no_nodes)