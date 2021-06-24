import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'bush_based_test_06'

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
pos = G.graph['pos']

tol = 10**-5
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
x =data['x'][-1]
w =data['weight'][-1]
print(x)
print(sorted(G.edges()))

print([(i,j,x,w) for ((i,j),x,w) in zip(sorted(G.edges()),x,w) if x > 0])
#print(data['objective'][-1])
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()

tapnx.update_edge_attribute(G, 'x', np.round(data['x'][-1],1))

fig, ax = tapnx.plot_graph(G, edge_labels=True, edge_label_attr='x')
plt.show()