import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def run_compare_methods(G, tol, max_iter):
    
    G, data_sa = tapnx.successive_averages(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
    G, data_fw = tapnx.frank_wolfe(G, aec_gap_tol=tol, collect_data=True, max_iter=max_iter)
    G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter)

    print(data['x'][-1])
    print(data['objective'][-1])
    #print('nq_measure = {}'.format(data['nq_measure']))
    #print('LM_measure = {}'.format(data['LM_measure']))

    plt.plot(data_sa['AEC'], label='Successive Averages')
    plt.plot(data_fw['AEC'], label='Frank Wolfe')
    plt.plot(data['AEC'], label='Gradient Projection')

    plt.xlabel('No. Iterations')
    plt.ylabel('AEC')
    plt.yscale('log')
    plt.legend()


    plt.figure()
    plt.plot(data_sa['relative_gap'], label='Successive Averages')
    plt.plot(data_fw['relative_gap'], label='Frank Wolfe')
    plt.plot(data['relative_gap'], label='Gradient Projection')
    plt.xlabel('No. Iterations')
    plt.ylabel('Relative GAP')
    plt.yscale('log')
    plt.legend()

    plt.show()

def run_gradient_projection(G, tol, max_iter):
    G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter, verbose=True)


if __name__ == "__main__":
    filename = 'siouxfalls'
    #filename = 'chicagosketch'
    #filename = 'anaheim'
    G = tapnx.graph_from_TNTP(filename, edge_attr=True)
    tol = 10**-5
    max_iter = 1000

    #run_gradient_projection(G, tol, max_iter)
    run_compare_methods(G, tol, max_iter)