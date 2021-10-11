import matplotlib.cm as cm
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import vander
import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np

#filename = 'siouxfalls'
#filename = 'grid_side_to_side_large'
#filename = 'grid_corner_to_corner_large'

for filename in ['grid_side_to_side_large', 'bush_based_test_06', 'siouxfalls', 'nq_example1']:

#for filename in ['siouxfalls_repositioned']:
    fig, ax = plt.subplots(figsize=(20,20))
    G = tapnx.graph_from_csv(filename, nodes=True, edge_attr=True)

    # load dataframe and get 'measure as list.
    # add measure to edge attributes
    # df_stats = pd.read_csv('test_data/{}/{}_edge_metrics.csv'.format(filename,filename))
    # recriprocalttnorm = df_stats['reciprocalttnorm'].tolist()
    # G = tapnx.utils_graph.update_edge_attribute(G, 'IR', recriprocalttnorm)

    # recriprocalttnorm_edge_labels_outgoing = {(u,v):np.round(ir,3) for ((u,v),ir) in zip(sorted(G.edges()), tapnx.utils_graph.get_np_array_from_edge_attribute(G,'IR')) if u<v}
    # recriprocalttnorm_edge_labels_incoming = {(u,v):np.round(ir,3) for ((u,v),ir) in zip(sorted(G.edges()), tapnx.utils_graph.get_np_array_from_edge_attribute(G,'IR')) if v<u}
    

    edge_list_outgoing = [(u,v) for (u,v) in G.edges() if u < v ]
    edge_list_incoming = [(u,v) for (u,v) in G.edges() if u > v ]

    xshift = 0.15
    yshift = 0.15
    xshift = 0
    yshift = 0
    offset_outgoing_pos = {k:(x+xshift,y+yshift) for k,(x,y) in G.graph['pos'].items()}
    offset_incoming_pos = {k:(x-xshift,y-yshift) for k,(x,y) in G.graph['pos'].items()}

    fig, ax = tapnx.plot_nodes(G, pos=G.graph['pos'], node_size=800, ax=ax, node_color='w', edgecolors='k')
    fig, ax = tapnx.plot_node_labels(G, G.graph['pos'], ax=ax, font_size=16)
    fig, ax = tapnx.plot_edges(G, pos=offset_outgoing_pos, edge_list=edge_list_outgoing, width=1.5, arrowsize=25, ax=ax,min_source_margin=0.1,edge_color='k')
    fig, ax = tapnx.plot_edges(G, pos=offset_incoming_pos, edge_list=edge_list_incoming, width=1.5, arrowsize=25, ax=ax,min_source_margin=0.1,edge_color='k')   

    ax.axis('off')
    
    #plt.show()
    plt.savefig('network_plot_{}.pdf'.format(filename), bbox_inches = 'tight')


