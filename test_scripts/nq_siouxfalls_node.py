import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def importance_computation(G,E,n,verbose=False):
    H = tapnx.remove_node(G,n)
    H, data = tapnx.gradient_projection(H, collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=verbose)
    return tapnx.importance_measure(E,E1)

filename = 'siouxfalls'

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tol = 10**-2
max_iter = 1000
verbose = False
G, data = tapnx.gradient_projection(G, collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose)
E = data['nq_measure']

importance_results = {}
importance_results = {}
nodes = list(sorted(G.nodes()))

for n in nodes:
    print('computing NQ for node {}'.format(n))
    importance_results[n] = np.round(importance_computation(G,E,n,verbose),4)

print(importance_results)

centrality_measures = nx.algorithms.centrality.betweenness_centrality(G)
closeness_measures = nx.algorithms.centrality.closeness_centrality(G)
deg_centrality_measures = nx.algorithms.centrality.degree_centrality(G)
harm_centrality_measures = nx.algorithms.centrality.harmonic_centrality(G)

print(centrality_measures)
plt.figure()
plt.plot([value for key, value in importance_results.items()][:-1], [value for key, value in centrality_measures.items()][:-1], 'o')

plt.figure()
plt.plot([value for key, value in importance_results.items()], [value for key, value in closeness_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results.items()], [value for key, value in deg_centrality_measures.items()], 'o')


plt.figure()
plt.plot([value for key, value in importance_results.items()], [value for key, value in harm_centrality_measures.items()], 'o')

plt.show()