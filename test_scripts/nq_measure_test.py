import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'nq_example'
filename = 'siouxfalls'

G = tapnx.graph_from_csv(filename, nodes=None, trips=True, edge_attr=True)

tol = 10**-4
max_iter = 100
G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
print(data)
print('nq_measure = {}'.format(data['nq_measure']))
print('LM_measure = {}'.format(data['LM_measure']))
