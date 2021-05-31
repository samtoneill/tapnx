import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


filename = 'amendedboyles190'
#filename = 'siouxfalls'


G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
    
tol = 10**-5

G, data = tapnx.successive_averages(G,aec_gap_tol=tol,max_iter=100,collect_data=True)
plt.plot(data['AEC'], label='Successive Averages')
G, data = tapnx.frank_wolfe(G,aec_gap_tol=tol,max_iter=100,collect_data=True)
plt.plot(data['AEC'], label='Frank Wolfe')
G, data = tapnx.gradient_projection(G,aec_gap_tol=tol,max_iter=100,collect_data=True)
plt.plot(data['AEC'], label='Gradient Projection')
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()
plt.show()