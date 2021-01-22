"""Graph utility functions."""

import networkx as nx
import numpy as np
import pandas as pd
import time

def edges_from_path(path):
    return list(zip(path,path[1:]))

def path_length(G, path):
    edges = edges_from_path(path)
    return sum([G[u][v]['weight'] for (u,v) in edges])

def graph_from_edgedf(df_edges, edge_attr=None):
    """
    Convert node and edge DataFrames to a DiGraph.
    Parameters
    ----------
    df_edges : Pandas dataframe
        DataFrame of edges
    edge_attr : Bool
        Include attributes in graph
    Returns
    -------
    G : networkx.DiGraph
        Networkx directed graph
    """
    # create a new networkx graph based on the edges dataframe edges_df
    G = nx.from_pandas_edgelist(df_edges, source='source', target='target',
                                 edge_attr=edge_attr, create_using=nx.DiGraph())
    G.graph['weight'] = 'weight'
    G.graph['pos'] = None
    G = _label_edges_with_id(G)
    return G

def graph_positions_from_nodedf(G, df_nodes):
    """
    Read node positional data from dataframe and store within the graph
    ----------
    G : networkx.DiGraph
        Networkx directed graph
    df_nodes : Pandas dataframe 
        DataFrame of node positions
    Returns
    -------
    G : networkx.DiGraph
        Networkx directed graph
    """
    # read and format the positional data
    pos = df_nodes.to_dict(orient='index')
    G.graph['pos'] = {id:(pos[id][x],pos[id][y]) for id, (x,y) in pos.items()}

    return G

def graph_edge_weight_func(G, edge_func, edge_func_integral):
    """
    Update the weight attribute of the graph. This is used to calculate the shortest path
    ----------
    G : networkx.DiGraph
        Network Graph
    weight : (string or function)
        If this is a string, then edge weights will be accessed via the edge attribute with this key (that is, the weight of the edge joining u to v will be G.edges[u, v][weight]). If no such edge attribute exists, the weight of the edge is assumed to be one.

        If this is a function, the weight of an edge is the value returned by the function. The function must accept exactly three positional arguments: the two endpoints of an edge and the dictionary of edge attributes for that edge. The function must return a number.
    Returns
    -------
    G : networkx.DiGraph
    """
    G.graph['edge_func'] = edge_func
    G.graph['edge_func_integral'] = edge_func_integral
    return G

def graph_trips_from_tripsdf(G, df_trips):
    """
    Covert trips martix to a dictionary and store in the graph
    ----------
    G : networkx.DiGraph
        Networkx directed graph
    df_trips : Pandas dataframe
        DataFrame of Origin/Destination trips
    Returns
    -------
    G : networkx.DiGraph
        Networkx directed graph
    """
    
    trips = df_trips.to_dict(orient='index')
    G.graph['trips'] = trips
    return G

def graph_from_csv(edges_filename, nodes_filename=None, trips_filename=None, edge_attr=False):
    df_edges = pd.read_csv(edges_filename, index_col=[0])
    G = graph_from_edgedf(df_edges, edge_attr=True)

    if nodes_filename:
        df_nodes = pd.read_csv(nodes_filename, index_col=[0])
        G = graph_positions_from_nodedf(G, df_nodes)
    
    if trips_filename:
        df_trips = pd.read_csv(trips_filename, index_col=0)
        df_trips.fillna(0, inplace=True)
        G = graph_trips_from_tripsdf(G, df_trips)

    return G

def _label_edges_with_id(G):
    for index, (u,v) in enumerate(sorted(G.edges(), key= lambda edge: (edge[0], edge[1]))):
        G[u][v]['id'] = index
    return G

def remove_edge(G, u, v):
    G.remove_edge(u,v)
    G = _label_edges_with_id(G)
    return G
