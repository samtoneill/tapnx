import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'pigou_counter'
filename = 'pigou_counter_1'

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)

edge_func = lambda x, a, b, c, n: a + b*(x/c)**n
edge_func_derivative = lambda x, a, b, c, n: (b*n/(c**n))*x**(n-1) 

tol = 10**-4
G, data_G = tapnx.gradient_projection(G, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol, verbose=True)

print(data_G['x'])
print(data_G['total_time'])

G = tapnx.remove_edge(G,3,2)

G, data_G = tapnx.gradient_projection(G, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol)

print(data_G['x'])
print(data_G['total_time'])
