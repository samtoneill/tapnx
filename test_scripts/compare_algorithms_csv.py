import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


filename = 'amendedboyles190'
filename = 'siouxfalls'


G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
    
tol = 10**-5
no_iter = 500

G, data_s = tapnx.successive_averages(G,aec_gap_tol=tol,max_iter=no_iter,collect_data=True)
G, data_f = tapnx.frank_wolfe(G,aec_gap_tol=tol,max_iter=no_iter,collect_data=True)
G, data_g = tapnx.gradient_projection(G,aec_gap_tol=tol,max_iter=no_iter,collect_data=True)
plt.plot(data_s['AEC'], label='Successive Averages')
plt.plot(data_f['AEC'], label='Frank Wolfe')
plt.plot(data_g['AEC'], label='Gradient Projection')
plt.xlabel('No. Iterations')
plt.ylabel('AEC')
plt.yscale('log')
plt.legend()

plt.figure()
plt.plot(data_s['time_taken'], label='Successive Averages')
plt.plot(data_f['time_taken'], label='Frank Wolfe')
plt.plot(data_g['time_taken'], label='Gradient Projection')
plt.xlabel('No. Iterations')
plt.ylabel('Time')
plt.legend()
plt.show()