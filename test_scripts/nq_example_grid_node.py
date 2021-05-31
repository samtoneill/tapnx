import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def importance_computation(G,E,n,measure,verbose=False):
    H = tapnx.remove_node(G,n)
    H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=verbose)
    E1 = data[measure]
    return tapnx.importance_measure(E,E1)

filename = 'nq_example_grid'

edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tol = 10**-2
max_iter = 1000
verbose = False
G, data = tapnx.gradient_projection(G,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose)
E_nq = data['nq_measure']
E_tt = data['nq_measure']

importance_results_nq = {}
importance_results_tt = {}
nodes = list(sorted(G.nodes()))

for n in nodes:
    print('computing NQ for node {}'.format(n))
    importance_results_nq[n] = np.round(importance_computation(G,E_nq,n,'nq_measure',verbose),4)
    importance_results_tt[n] = np.round(importance_computation(G,E_tt,n,'total_time',verbose),4)



print(importance_results_nq)
print(importance_results_tt)

centrality_measures = nx.algorithms.centrality.betweenness_centrality(G)
closeness_measures = nx.algorithms.centrality.closeness_centrality(G)
deg_centrality_measures = nx.algorithms.centrality.degree_centrality(G)
plt.figure()
plt.plot([value for key, value in importance_results_nq.items()][:-1], [value for key, value in centrality_measures.items()][:-1], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_nq.items()], [value for key, value in closeness_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_nq.items()], [value for key, value in deg_centrality_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_tt.items()], [value for key, value in centrality_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_tt.items()], [value for key, value in closeness_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_tt.items()], [value for key, value in deg_centrality_measures.items()], 'o')

plt.figure()
plt.plot([value for key, value in importance_results_nq.items()], [value for key, value in importance_results_tt.items()], 'o')

plt.show()

