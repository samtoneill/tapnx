import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'braess_wiki'

G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)

edge_func = lambda x, a, b, c, n: a + b*(x/c)**n
edge_func_derivative = lambda x, a, b, c, n: (b*n/(c**n))*x**(n-1) 

tol = 10**-5
G, data_G = tapnx.gradient_projection(G, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol)

print(data_G['x'])
print(data_G['total_time'])

H = tapnx.remove_edge(G,2,3)

H, data_H = tapnx.gradient_projection(H, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol)

print(data_H['x'])


print('total time removing (2,3) = {}'.format(data_H['total_time']))

print('total time metric removing (2,3) = {}'.format((data_H['total_time']-data_G['total_time'])/data_H['total_time']))

print('nq removing (2,3) = {}'.format((data_G['nq_measure']-data_H['nq_measure'])/data_G['nq_measure']))

H = tapnx.remove_edge(G,1,2)

H, data_H = tapnx.gradient_projection(H, edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol)

print(data_H['x'])

print('total time removing (1,2) = {}',format(data_H['total_time']))

print('total time metric removing (1,2) = {}'.format((data_H['total_time']-data_G['total_time'])/data_H['total_time']))

print('nq removing (1,2) = {}'.format((data_G['nq_measure']-data_H['nq_measure'])/data_G['nq_measure']))
