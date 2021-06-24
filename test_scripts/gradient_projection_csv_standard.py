import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

filename = 'chicagosketch'

G = tapnx.graph_from_csv(filename, nodes=False, trips=True, edge_attr=True)


tol = 10**-4
G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,verbose=True, max_iter=2)

# plt.plot(data['AEC'], label='Gradient Projection')
# plt.xlabel('No. Iterations')
# plt.ylabel('AEC')
# plt.yscale('log')
# plt.legend()
# plt.show()

print(data['x'][-1])