import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


filename = 'amendedboyles190'
G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
G.graph['no_edges']= G.number_of_edges()
#filename = 'siouxfalls'
#G = tapnx.graph_from_TNTP(filename, nodes=True, edge_attr=True)


fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
plt.show()
tapnx.milp_tap(G)