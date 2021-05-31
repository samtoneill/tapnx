import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def run_compare_methods(G, tol, max_iter):
    
    G, data_sa = tapnx.successive_averages(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
    plt.plot(data_sa['AEC'], label='Successive Averages')
    G, data_fw = tapnx.frank_wolfe(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
    plt.plot(data_fw['AEC'], label='Frank Wolfe')
    G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
    plt.plot(data['AEC'], label='Gradient Projection 1')

    print(data['x'][-1])
    print(data['objective'][-1])
    print('nq_measure = {}'.format(data['nq_measure']))
    print('LM_measure = {}'.format(data['LM_measure']))
    plt.xlabel('No. Iterations')
    plt.ylabel('AEC')
    plt.yscale('log')
    plt.legend()

    plt.figure()
    plt.plot(data['no_paths'], data['AEC'])
    plt.yscale('log')

    plt.show()

def run_gradient_projection(G, tol, max_iter):
    G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=True)


if __name__ == "__main__":
    filename = 'siouxfalls'
    filename = 'chicagosketch'
    #filename = 'anaheim'
    G = tapnx.graph_from_TNTP(filename, edge_attr=True)
    tol = 10**-5
    max_iter = 100

    run_gradient_projection(G, tol, max_iter)