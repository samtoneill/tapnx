from networkx.algorithms.centrality.betweenness import betweenness_centrality
from networkx.algorithms.centrality.degree_alg import degree_centrality
import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import rankdata

def compute_edge_metrics(filename, edge_func, edge_func_derivative, tol=10**-2, max_iter=100,verbose=False,alpha=1):
    G = tapnx.graph_from_csv(filename, nodes=False, trips=True, edge_attr=True)
    
    # compute centrality measures before loading network. We want to examine weighted paths on an empty network
    


    #edge_betweenness_od = nx.algorithms.centrality.edge_betweenness_centrality_subset(G, sources = [1], targets=[19,20], normalized=True)
    #edge_betweenness_od = [value for key, value in sorted(edge_betweenness_od.items())]


    G, data_G = tapnx.gradient_projection(G,edge_func=edge_func,edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose,alpha=alpha)
    x = data_G['x'][-1]
    weight = data_G['weight'][-1]
    trips = G.graph['trips']
    
    measures = ['nq_measure', 'nq_measure_norm','LM_measure', 
                'ITT','system_optimal', 'zhu_measure', 'zhu_measure_norm', 
                'reciprocal_system_optimal', 'reciprocal_tt', 
                'reciprocal_tt_norm', 'unsatisfied_demand', 'global_importance',
                'demand_weighted_importance', 'path_based_tt']
    Is_edges = {}
    importance_results_edges = {}
        
    for (u,v) in sorted(G.edges()):
        print('computing importance measures for edge ({},{})'.format(u,v))
        H = tapnx.remove_edge(G, u,v)
        H, data_H = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose,alpha=alpha)
        
        for measure in measures:
            print(measure)
            E = data_G[measure]
            E1 = data_H[measure]
            if measure == 'nq_measure':
                print('E_NQ......\n\n')
                print(E)
                print(E1)
            # for the total travel time we compute this the opposite to get a number been 0 and 1
            if measure in ['reciprocal_tt_norm', 'ITT', 'zhu_measure_norm', 'nq_measure_norm']:
                importance_results_edges[(u,v,measure)] = tapnx.importance_measure(E,E1,use_max=True)
            elif measure in ['zhu_measure', 'global_importance']:
                importance_results_edges[(u,v,measure)] = tapnx.importance_measure(E,E1,zhu=True)
            elif measure == 'unsatisfied_demand':
                importance_results_edges[(u,v,measure)] = data_H[measure]
            elif measure == 'demand_weighted_importance':
                importance_results_edges[(u,v,measure)] = tapnx.importance_measure(E,E1,demand_weighted_importance=True)
            elif measure == 'path_based_tt':
                importance_results_edges[(u,v,'total_time_difference')] = E1-E
            else:
                importance_results_edges[(u,v,measure)] = tapnx.importance_measure(E,E1,use_max=False)

    
            
        importance_results_edges[(u,v,'system_optimal')] = data_H['system_optimal']
        importance_results_edges[(u,v,'path_based_tt')] = data_H['path_based_tt']
    
    Is_nodes = {}
    importance_results_nodes = {}
        
    for n in sorted(G.nodes()):
        print('computing importance measures for node {}'.format(n))
        H = tapnx.remove_node(G, n)
        H, data_H = tapnx.gradient_projection(H,edge_func=edge_func, edge_func_derivative=edge_func_derivative,collect_data=True,aec_gap_tol=tol,max_iter=max_iter,verbose=verbose,alpha=alpha)
        for measure in measures:
            print(measure)
            # for the total travel time we compute this the opposite to get a number been 0 and 1
            E = data_G[measure]
            E1 = data_H[measure]
            print(E)
            print(E1)
            if measure in ['reciprocal_tt_norm', 'ITT', 'zhu_measure_norm', 'nq_measure_norm']:
                print('orginal TT = {}'.format(E1))
                print('new TT = {}'.format(E))
                print('Diff = {}'.format(E1-E))
                importance_results_nodes[(n,measure)] = tapnx.importance_measure(E,E1,use_max=True)
            # note that the zhu measure is defined in Nagurney  as (E1-E)/E
            elif measure in ['zhu_measure', 'global_importance']:
                importance_results_nodes[(n,measure)] = tapnx.importance_measure(E,E1,zhu=True) 
            elif measure == 'unsatisfied_demand':
                importance_results_nodes[(n,measure)] = data_H[measure]
            elif measure == 'demand_weighted_importance':
                importance_results_nodes[(n,measure)] = tapnx.importance_measure(E,E1,demand_weighted_importance=True)
            elif measure == 'path_based_tt':
                importance_results_nodes[(n,'total_time_difference')] = E1-E
            else:
                importance_results_nodes[(n,measure)] = tapnx.importance_measure(E,E1,use_max=False)
        
        importance_results_nodes[(n,'system_optimal')] = data_H['system_optimal']
        importance_results_nodes[(n,'path_based_tt')] = data_H['path_based_tt']

    for measure in measures:
        Is_edges[measure] = [value for (u,v,m),value in sorted(importance_results_edges.items()) if m == measure]
        Is_nodes[measure] = [value for (n,m),value in sorted(importance_results_nodes.items()) if m == measure]

    Is_edges['total_time_difference'] = [value for (u,v,m),value in sorted(importance_results_edges.items()) if m == 'total_time_difference']
    Is_nodes['total_time_difference'] = [value for (n,m),value in sorted(importance_results_nodes.items()) if m == 'total_time_difference']
    
    u = [u for (u,v) in sorted(G.edges())]
    v = [v for (u,v) in sorted(G.edges())]
    n = [n for n in sorted(G.nodes())]

    
    #trips_adjacent = [trips[u]['{}'.format(v)] for (u,v) in sorted(G.edges())]
    trips_adjacent = [np.sum([value for key, value in trips[u].items()]) + np.sum([value for key, value in trips[v].items()]) for (u,v) in sorted(G.edges())]
    x_adjacent_node = [np.sum([xe for (ue,ve,xe) in zip(u,v,x) if node == ue or node == ve]) for node in n]

    edge_betweenness = nx.algorithms.centrality.edge_betweenness_centrality(G)
    edge_betweenness = [value for key, value in sorted(edge_betweenness.items())]

    edge_betweenness_w = nx.algorithms.centrality.edge_betweenness_centrality(G, weight='weight')
    edge_betweenness_w = [value for key, value in sorted(edge_betweenness_w.items())]

    weighted_clustering_coeff = nx.algorithms.cluster.clustering(G)
    weighted_clustering_coeff = [value for key, value in sorted(weighted_clustering_coeff.items())]

    weighted_clustering_coeff_w = nx.algorithms.cluster.clustering(G, weight='weight')
    weighted_clustering_coeff_w = [value for key, value in sorted(weighted_clustering_coeff_w.items())]

    closeness_centrality = nx.algorithms.centrality.closeness_centrality(G)
    closeness_centrality = [value for key, value in sorted(closeness_centrality.items())]

    closeness_centrality_w = nx.algorithms.centrality.closeness_centrality(G, distance='weight')
    closeness_centrality_w = [value for key, value in sorted(closeness_centrality_w.items())]

    degree_centrality = nx.algorithms.centrality.degree_centrality(G)
    degree_centrality =  [value for key, value in sorted(degree_centrality.items())]

    betweenness = nx.algorithms.centrality.betweenness_centrality(G)
    betweenness = [value for key, value in sorted(betweenness.items())]

    betweenness_w = nx.algorithms.centrality.betweenness_centrality(G, weight='weight')
    betweenness_w = [value for key, value in sorted(betweenness_w.items())]
    

    a = tapnx.utils_graph.get_np_array_from_edge_attribute(G, 'a')
    G = tapnx.utils_graph.update_edge_attribute(G, 'weight',a) 

    edge_betweenness_a = nx.algorithms.centrality.edge_betweenness_centrality(G, weight='a')
    edge_betweenness_a = [value for key, value in sorted(edge_betweenness_a.items())]
    weighted_clustering_coeff_a = nx.algorithms.cluster.clustering(G, weight='a')
    weighted_clustering_coeff_a = [value for key, value in sorted(weighted_clustering_coeff_a.items())]
    closeness_centrality_a = nx.algorithms.centrality.closeness_centrality(G, distance='a')
    closeness_centrality_a = [value for key, value in sorted(closeness_centrality_a.items())]
    betweenness_a = nx.algorithms.centrality.betweenness_centrality(G, weight='a')
    betweenness_a = [value for key, value in sorted(betweenness_a.items())]

    print(edge_betweenness)
    print(edge_betweenness_w)
    print(edge_betweenness_a)
    print(degree_centrality)

    # this needs updating
    #sources = [origin for origin, destinations in trips.items()]
    #targets = [origin for origin, destinations in trips.items()]

    #edge_betweenness_od_w = nx.algorithms.centrality.edge_betweenness_centrality_subset(G, sources = [7], targets=[3], normalized=True, weight='weight')
    #edge_betweenness_od_w = [value for key, value in sorted(edge_betweenness_od_w.items())]

    cap = tapnx.utils_graph.get_np_array_from_edge_attribute(G, 'c')

    results_edges = {'source':u, 'target':v, 
                'x':x, 
                'xrank': rankdata(-1*np.round(x,4), method='dense'),
                'INQ':Is_edges['nq_measure'], 
                'INQrank': rankdata(-1*np.round(np.array(Is_edges['nq_measure']),4), method='dense'),
                'INQnorm':Is_edges['nq_measure_norm'], 
                'ILM':Is_edges['LM_measure'], 
                'ILMrank': rankdata(-1*np.array(Is_edges['LM_measure']), method='dense'),
                'ITT':Is_edges['ITT'], 
                'IZ':Is_edges['zhu_measure'],
                'IZrank': rankdata(-1*np.array(Is_edges['zhu_measure']), method='dense'),
                'IZnorm':Is_edges['zhu_measure_norm'],
                'tripsadj':trips_adjacent, 
                'weight':weight,
                'reciprocalsystemoptimal':Is_edges['reciprocal_system_optimal'],
                'reciprocaltt':Is_edges['reciprocal_tt'],
                'reciprocalttnorm':Is_edges['reciprocal_tt_norm'],
                'reciprocalttnormrank': rankdata(-1*np.array(Is_edges['reciprocal_tt_norm']), method='dense'),
                'unsatisfieddemand':Is_edges['unsatisfied_demand'],
                'unsatisfieddemandrank': rankdata(-1*np.array(Is_edges['unsatisfied_demand']), method='dense'),
                'globalimportance':Is_edges['global_importance'],
                'demandweightedimportance':Is_edges['demand_weighted_importance'],
                'totaltimedifference':Is_edges['total_time_difference'],
                'totaltimedifferencerank': rankdata(-1*np.array(Is_edges['total_time_difference']), method='dense'),
                'systemoptimal':Is_edges['system_optimal'],
                'pathbasedtt':Is_edges['path_based_tt'],
                'edge_betweenness':edge_betweenness,
                'edge_betweenness_w':edge_betweenness_w,
                'edge_betweenness_a':edge_betweenness_a
                }

    df_edges = pd.DataFrame.from_dict(results_edges)

    df_edges['x_fraction'] = df_edges['x']/df_edges['x'].sum()
    df_edges['weight_fraction'] = df_edges['weight']/df_edges['weight'].sum()
    
    results_nodes = {'node':n, 
                'INQ':Is_nodes['nq_measure'], 
                'INQrank':[x if not np.isnan(Is_nodes['nq_measure'][i]) else np.nan for i,x in enumerate(rankdata(-1*np.array(Is_nodes['nq_measure']), method='dense'))],
                'INQnorm':Is_nodes['nq_measure_norm'], 
                'ILM':Is_nodes['LM_measure'], 
                'ILMrank':[x if not np.isnan(Is_nodes['LM_measure'][i]) else np.nan for i,x in enumerate(rankdata(-1*np.array(Is_nodes['LM_measure']), method='dense'))],
                'ITT':Is_nodes['ITT'], 
                'IZ':Is_nodes['zhu_measure'],
                'IZrank':[x if not np.isnan(Is_nodes['zhu_measure'][i]) else np.nan for i,x in enumerate(rankdata(-1*np.array(Is_nodes['zhu_measure']), method='dense'))],
                'IZnorm':Is_nodes['zhu_measure_norm'],
                'reciprocalsystemoptimal':Is_nodes['reciprocal_system_optimal'],
                'reciprocaltt':Is_nodes['reciprocal_tt'],
                'reciprocalttnorm':Is_nodes['reciprocal_tt_norm'],
                'reciprocalttnormrank':[x if not np.isnan(Is_nodes['reciprocal_tt_norm'][i]) else np.nan for i,x in enumerate(rankdata(-1*np.array(Is_nodes['reciprocal_tt_norm']), method='dense'))],
                'unsatisfieddemand':Is_nodes['unsatisfied_demand'],
                'unsatisfieddemandrank': rankdata(-1*np.array(Is_nodes['unsatisfied_demand']), method='dense'),
                'globalimportance':Is_nodes['global_importance'],
                'demandweightedimportance':Is_nodes['demand_weighted_importance'],
                'totaltimedifference':Is_nodes['total_time_difference'],
                'totaltimedifferencerank': rankdata(-1*np.array(Is_nodes['total_time_difference']), method='dense'),
                'systemoptimal':Is_nodes['system_optimal'],
                'pathbasedtt':Is_nodes['path_based_tt'],
                'xadj':x_adjacent_node,
                'xadjrank': rankdata(-1*np.array(x_adjacent_node), method='dense'),
                'weighted_clustering_coeff': weighted_clustering_coeff,
                'weighted_clustering_coeff_w': weighted_clustering_coeff_w,
                'weighted_clustering_coeff_a': weighted_clustering_coeff_a,
                'closeness_centrality':closeness_centrality,
                'closeness_centrality_w':closeness_centrality_w,
                'closeness_centrality_a':closeness_centrality_a,
                'degree_centrality':degree_centrality,
                'betweenness':betweenness,
                'betweenness_w':betweenness_w,
                'betweenness_a':betweenness_a
                }
    df_nodes = pd.DataFrame.from_dict(results_nodes)

    
    return G, df_edges, df_nodes


if __name__ == "__main__":
    alpha = 1
    filename = 'siouxfalls'
    #filename = 'anaheim'
    alpha = 0.5
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

    # filename = 'nq_example'    
    # edge_func = lambda x, a, b, c, n: a + b*x + c*x**n
    # edge_func_derivative = lambda x, a, b, c, n: b + c*n*x**(n-1)

    # filename = 'braess_wiki'    
    # edge_func = lambda x, a, b, c, n: a + b*(x/c)**n
    # edge_func_derivative = lambda x, a, b, c, n: (b*n/(c**n))*x**(n-1) 

    # filename = 'grid_corner_to_corner'    
    # filename = 'grid_side_to_side'   
    # filename = 'grid_side_to_side_large_reduced_demand'   
    # filename = 'grid_side_to_side_large_increased_demand'   
    # filename = 'grid_side_to_side_large'   
    # # filename = 'grid_corner_to_corner_large'   
    # alpha = 0.1
    # # # # set alpha to 0.2
    # edge_func = lambda x, a, b, c, n: a + b*(x/c)**n
    # edge_func_derivative = lambda x, a, b, c, n: (b*n/(c**n))*x**(n-1) 
    
    tol = 10**-5
    max_iter = 2000
    G, df_edges, df_nodes = compute_edge_metrics(filename, edge_func, edge_func_derivative, tol=tol, max_iter=max_iter,verbose=False,alpha=alpha)
    df_edges.to_csv('test_data/{}/{}_edge_metrics.csv'.format(filename,filename), na_rep='N/A',  float_format='%.5f')
    df_nodes.to_csv('test_data/{}/{}_node_metrics.csv'.format(filename,filename), na_rep='N/A',  float_format='%.5f')

    # tapnx.plot_graph(G)
    # plt.show()

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
