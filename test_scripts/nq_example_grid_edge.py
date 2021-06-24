import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def importance_computation(G,E,u,v):
    H = tapnx.remove_edge(G, u,v)
    H, data = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter,alpha=0.5)
    E1 = data['nq_measure']
    return tapnx.importance_measure(E,E1)

filename = 'nq_example_grid'

edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
# fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
# plt.show()
tol = 10**-7
max_iter = 1000
G, data = tapnx.gradient_projection(G,edge_func=edge_func, edge_func_derivative=edge_func_derivative, collect_data=True,aec_gap_tol=tol,max_iter=max_iter,alpha=0.5)
E = data['nq_measure']

importance_results = {}
for (u,v) in sorted(G.edges()):
    print('computing NQ for edge ({},{})'.format(u,v))
    importance_results[(u,v)] = np.round(importance_computation(G,E,u,v),4)

# compute the NQ measure as labelled in Nagurney 07
edge_names = {
    (1,2):1,
    (2,3):2,
    (3,4):3,
    (4,5):4,
    (5,6):5,
    (6,7):6,
    (7,8):7,
    (8,9):8,
    (9,10):9,
    (1,11):10,
    (2,12):11,
    (3,13):12,
    (4,14):13,
    (5,15):14,
    (6,16):15,
    (7,17):16,
    (8,18):17,
    (9,19):18,
    (10,20):19,
    (11,12):20,
    (12,13):21,
    (13,14):22,
    (14,15):23,
    (15,16):24,
    (16,17):25,
    (17,18):26,
    (18,19):27,
    (19,20):28,
}

nagurney_labelling = {edge_names[key]: value for key, value in sorted(importance_results.items())}
for key, value in sorted(nagurney_labelling.items(), key=lambda item: item[0], reverse=False):
    print(key, value)

# compute the 

I = np.array([value for key, value in sorted(importance_results.items())])

G = tapnx.update_edge_attribute(G, 'importance', I)

edge_color = tapnx.get_edge_colors_by_attr(G, 'importance')

# plot network with edges coloured by edge attribute value
fig, ax = tapnx.plot_graph(
    G, edge_color=edge_color, node_size=200, node_labels=True,
    edge_labels=True, edge_label_attr='importance'
)
fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)

plt.show()