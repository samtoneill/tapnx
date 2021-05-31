import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def importance_computation(G,E,n):
    H = tapnx.remove_node(G,n)
    H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=False)
    E1 = data['nq_measure']
    print(data['x'][-1])
    print(E)
    print(E1)
    return tapnx.importance_measure(E,E1)

filename = 'nq_bad'

edge_func = lambda x, a, b, c, n:  b*x 
edge_func_derivative = lambda x, a, b, c, n: b 

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tol = 10**-2
max_iter = 1000
G, data = tapnx.gradient_projection(G,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=False)
E = data['nq_measure']

importance_results = {}
nodes = list(sorted(G.nodes()))
for n in nodes:
    print('computing NQ for node {}'.format(n))
    print(data['x'][-1])
    print(data)
    importance_results[n] = np.round(importance_computation(G,E,n),4)

print(importance_results)
