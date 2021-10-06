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

for filename in ['grid_side_to_side_large', 'grid_corner_to_corner_large', 'siouxfalls', 
                'grid_side_to_side_large_reduced_demand',
                'grid_side_to_side_large_increased_demand']:
    
    fig, ax = plt.subplots(figsize=(20,15))
    G = tapnx.graph_from_csv(filename, nodes=True, edge_attr=True)

    # load dataframe and get 'measure as list.
    # add measure to edge attributes
    df_stats = pd.read_csv('test_data/{}/{}_edge_metrics.csv'.format(filename,filename))
    recriprocalttnorm = df_stats['reciprocalttnorm'].tolist()
    G = tapnx.utils_graph.update_edge_attribute(G, 'IR', recriprocalttnorm)

    recriprocalttnorm_edge_labels_outgoing = {(u,v):np.round(ir,3) for ((u,v),ir) in zip(sorted(G.edges()), tapnx.utils_graph.get_np_array_from_edge_attribute(G,'IR')) if u<v}
    recriprocalttnorm_edge_labels_incoming = {(u,v):np.round(ir,3) for ((u,v),ir) in zip(sorted(G.edges()), tapnx.utils_graph.get_np_array_from_edge_attribute(G,'IR')) if v<u}
    

    edge_color = tapnx.get_edge_colors_by_attr(G, 'IR', cmap='Blues')
    edges_and_colors = list(zip(G.edges(), edge_color,recriprocalttnorm))
    edge_list_outgoing = [(u,v) for (u,v) in G.edges() if u < v ]
    edge_color_outgoing = [col for (u,v), col, IR in edges_and_colors if u < v ]
    edge_list_incoming = [(u,v) for (u,v) in G.edges() if u > v ]
    edge_color_incoming = [col for (u,v), col, IR in edges_and_colors if u > v ]

    shift = 0.1
    offset_outgoing_pos = {k:(x+shift,y+shift) for k,(x,y) in G.graph['pos'].items()}
    offset_incoming_pos = {k:(x-shift,y-shift) for k,(x,y) in G.graph['pos'].items()}

    fig, ax = tapnx.plot_nodes(G, pos=G.graph['pos'], node_size=300, ax=ax)
    fig, ax = tapnx.plot_node_labels(G, G.graph['pos'], ax=ax)
    fig, ax = tapnx.plot_edges(G, pos=offset_outgoing_pos, edge_color=edge_color_outgoing, edge_list=edge_list_outgoing, width=1.5, arrowsize=25, ax=ax)
    fig, ax = tapnx.plot_edges(G, pos=offset_incoming_pos, edge_color=edge_color_incoming, edge_list=edge_list_incoming, width=1.5, arrowsize=25, ax=ax)   

    shift = 0.5
    offset_outgoing_pos = {k:(x+shift,y+shift) for k,(x,y) in G.graph['pos'].items()}
    offset_incoming_pos = {k:(x-shift,y-shift) for k,(x,y) in G.graph['pos'].items()}

    fig, ax = tapnx.plot_edge_labels(G, offset_outgoing_pos, ax=ax, edge_labels=recriprocalttnorm_edge_labels_outgoing, font_size=20)
    fig, ax = tapnx.plot_edge_labels(G, offset_incoming_pos, ax=ax, edge_labels=recriprocalttnorm_edge_labels_incoming, font_size=20)
    ax.axis('off')
    print(filename)

    plt.savefig('colour_plot_{}.pdf'.format(filename), bbox_inches = 'tight')


