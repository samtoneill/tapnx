import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def run_compare_methods(G, tol, max_iter):
    
    for alpha in [x/10 for x in range(0,11,2)]:
    #for alpha in [1]:
        G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,alpha = alpha)
        plt.plot(data['AEC'], label=r"$\alpha$={}".format(alpha))

    plt.xlabel('No. Iterations')
    plt.ylabel('AEC')
    plt.yscale('log')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filename = 'bush_based_test_06'
    #filename = 'chicagosketch'
    #filename = 'anaheim'
    #G = tapnx.graph_from_TNTP(filename, edge_attr=True)
    G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
    tol = 10**-5
    max_iter = 500

    #run_gradient_projection(G, tol, max_iter)
    run_compare_methods(G, tol, max_iter)