import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#filename = 'siouxfalls'
#filename = 'chicagosketch'
filename = 'anaheim'

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    #nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)

G = tapnx.igraph_from_edgedf(df_edges, edge_attr=True)
print(G.get_edgelist())
for idx, e in enumerate(G.es):
    print(idx, e.source, e.target)