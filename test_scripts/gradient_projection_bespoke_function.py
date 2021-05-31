import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'boyles190'

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)

edge_func = lambda x, a, b, c, n: a + b*(x/c)**n
edge_func_derivative = lambda x, a, b, c, n: (b*n/(c**n))*x**(n-1) 

tol = 10**-5
G, data = tapnx.gradient_projection(G, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol)
plt.plot(data['AEC'], label='Gradient Projection')
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()
plt.show()