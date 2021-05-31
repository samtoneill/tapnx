import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def importance_computation_edge(G,E,u,v):
    H = tapnx.remove_edge(G, u,v)
    H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
    E1 = data['nq_measure']
    return tapnx.importance_measure(E,E1)

def importance_computation_node(G,E,n):
    H = tapnx.remove_node(G,n)
    H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=True)
    E1 = data['nq_measure']
    print(E)
    print(E1)
    return tapnx.importance_measure(E,E1)

filename = 'nq_example1'

edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tol = 10**-4
max_iter = 100
G, data = tapnx.gradient_projection(G,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=True)
E = data['nq_measure']

edge_names = {(0,1):'a', (0,2):'b', (1,3):'c', (1,4):'d', (2,3):'e', (2,4): 'f'}

for (u,v) in sorted(G.edges()):
    print('computing NQ for edge {}'.format(edge_names[(u-1,v-1)]))
    print(importance_computation_edge(G,E,u,v))

importance_results = {}
nodes = list(sorted(G.nodes()))
for n in nodes:
    print('computing NQ for node {}'.format(n))
    importance_results[n] = np.round(importance_computation_node(G,E,n),2)

print(importance_results)

# create a copy with the edge remove. Keep orginal reference to graph 
# H = tapnx.remove_edge(G, 1,2)

# fig, ax = tapnx.plot_graph(H, node_size=200, node_labels=True)
# plt.show()

# H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
# E1 = data['nq_measure']

# print(tapnx.importance_measure(E,E1))

# # create a copy with the edge remove. Keep orginal reference to graph 
# H1 = tapnx.remove_edge(G, 2,3)

# fig, ax = tapnx.plot_graph(H1, node_size=200, node_labels=True)
# plt.show()

# H1, data = tapnx.gradient_projection(H1,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
# E2 = data['nq_measure']

# print(tapnx.importance_measure(E,E2))

