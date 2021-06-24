import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing

def compute_edge_metrics(filename, edge_func, edge_func_derivative, tol=10**-2, max_iter=100,verbose=False,alpha=1):
    G = tapnx.graph_from_csv(filename, nodes=True, trips=True, edge_attr=True)
    
    # compute centrality measures before loading network. We want to examine weighted paths on an empty network
    edge_betweenness = nx.algorithms.centrality.edge_betweenness_centrality(G)
    edge_betweenness = [value for key, value in sorted(edge_betweenness.items())]

    edge_betweenness_od = nx.algorithms.centrality.edge_betweenness_centrality_subset(G, sources = [1], targets=[19,20], normalized=True)
    edge_betweenness_od = [value for key, value in sorted(edge_betweenness_od.items())]


    G, data_G = tapnx.gradient_projection(G,edge_func=edge_func,edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose,alpha=alpha)
    x = data_G['x'][-1]
    weight = data_G['weight'][-1]
    trips = G.graph['trips']
    
    measures = ['nq_measure','LM_measure', 'total_time']
    Is = {}
    importance_results = {}
        
    for (u,v) in sorted(G.edges()):
        print('computing importance measures for edge ({},{})'.format(u,v))
        H = tapnx.remove_edge(G, u,v)
        H, data_H = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose,alpha=alpha)
        for measure in measures:
            # for the total travel time we compute this the opposite to get a number been 0 and 1
            if measure == 'total_time':
                E1 = data_G[measure]
                E = data_H[measure]
                print('orginal TT = {}'.format(E1))
                print('new TT = {}'.format(E))
                print('Diff = {}'.format(E1-E))
            else:
                E = data_G[measure]
                E1 = data_H[measure]
            importance_results[(u,v,measure)] = np.round(tapnx.importance_measure(E,E1),4)
    
    for measure in measures:
        Is[measure] = [value for (u,v,m),value in sorted(importance_results.items()) if measure in (u,v,m)]
    
    u = [u for (u,v) in sorted(G.edges())]
    v = [v for (u,v) in sorted(G.edges())]

 
    #trips_adjacent = [trips[u]['{}'.format(v)] for (u,v) in sorted(G.edges())]
    trips_adjacent = [np.sum([value for key, value in trips[u].items()]) + np.sum([value for key, value in trips[v].items()]) for (u,v) in sorted(G.edges())]

    edge_betweenness_w = nx.algorithms.centrality.edge_betweenness_centrality(G, weight='weight')
    edge_betweenness_w = [value for key, value in sorted(edge_betweenness_w.items())]

    # this needs updating
    #sources = [origin for origin, destinations in trips.items()]
    #targets = [origin for origin, destinations in trips.items()]

    edge_betweenness_od_w = nx.algorithms.centrality.edge_betweenness_centrality_subset(G, sources = [7], targets=[3], normalized=True, weight='weight')
    edge_betweenness_od_w = [value for key, value in sorted(edge_betweenness_od_w.items())]

    cap = tapnx.utils_graph.get_np_array_from_edge_attribute(G, 'c')

    results = {'source':u, 'target':v, 'x':x, 'I_NQ':Is['nq_measure'], 'I_LM':Is['LM_measure'], 
                'I_TT':Is['total_time'], 'trips_adj':trips_adjacent, 'weight':weight,
                'edge_betweenness':edge_betweenness, 'edge_betweenness_od':edge_betweenness_od,
                'edge_betweenness_w':edge_betweenness_w, 'edge_betweenness_od_w':edge_betweenness_od_w
                }

    df = pd.DataFrame.from_dict(results)

    df['x_fraction'] = df['x']/df['x'].sum()
    df['weight_fraction'] = df['weight']/df['weight'].sum()

    
    return G, df


if __name__ == "__main__":
    filename = 'siouxfalls'
    filename = 'bush_based_test_06'
    #filename = 'nq_grid_corner_to_corner'
    #filename = 'nq_grid_side_to_side'
    edge_func = lambda x, a, b, c, n: a*(1 + b*(x/c)**n)
    edge_func_derivative = lambda x, a, b, c, n: (a*b*n*x**(n-1))/(c**n)
    
   
    # filename = 'nq_example_grid'
    # edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
    # edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)
    
    # filename = 'nq_example_grid_ext'
    # edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
    # edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)
    

    # filename = 'nq_example1'    
    # edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
    # edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)
    
    tol = 10**-4
    max_iter = 5000
    G, df = compute_edge_metrics(filename, edge_func, edge_func_derivative, tol=tol, max_iter=max_iter,verbose=False,alpha=0.5)
    df.to_csv('test_data/{}/{}_metrics.csv'.format(filename,filename))

    tapnx.plot_graph(G)
    plt.show()

# G = tapnx.update_edge_attribute(G, 'importance', I)

# edge_color = tapnx.get_edge_colors_by_attr(G, 'importance')

# # plot network with edges coloured by edge attribute value
# fig, ax = tapnx.plot_graph(
#     G, edge_color=edge_color, node_size=200, node_labels=True,
#     edge_labels=True, edge_label_attr='importance'
# )
# fig.colorbar(cm.ScalarMappable(norm=None, cmap='plasma'), ax=ax)

# cap = tapnx.utils_graph.get_np_array_from_edge_attribute(G, 'c')
# plt.figure()
# plt.plot(I, cap, 'o')

# edges = sorted(G.edges())
# trips = G.graph['trips']
# trips_adjacent = [trips[u]['{}'.format(v)] for (u,v) in edges]
# plt.figure()
# plt.plot(I, trips_adjacent, 'o')

# plt.figure()
# plt.plot(I, trips_adjacent/cap, 'o')

# plt.figure()
# plt.plot(I, x, 'o')


# plt.figure()
# plt.plot(I, x/cap, 'o')

# plt.show()