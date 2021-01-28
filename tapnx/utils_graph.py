"""Graph utility functions."""

import networkx as nx
from igraph import Graph
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
    G.graph['no_edges'] = len(G.edges())
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

def trips_from_tripsdf(G, df_trips):
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

def graph_from_csv(edges_filename, nodes_filename=None, trips_filename=None, edge_attr=False, igraph=False):



    df_edges = pd.read_csv(edges_filename, index_col=[0])
    df_nodes = None
    df_trips = None
    #G = graph_from_edgedf(df_edges, edge_attr=True)

    if nodes_filename:
        df_nodes = pd.read_csv(nodes_filename, index_col=[0])
    #    G = graph_positions_from_nodedf(G, df_nodes)
    
    if trips_filename:
        df_trips = pd.read_csv(trips_filename, index_col=0)
        df_trips.fillna(0, inplace=True)
    #    G = graph_trips_from_tripsdf(G, df_trips)

    return df_edges, df_nodes, df_trips

def _label_edges_with_id(G):
    for index, (u,v) in enumerate(sorted(G.edges(), key= lambda edge: (edge[0], edge[1]))):
        G[u][v]['id'] = index
    return G

def get_np_array_from_edge_attribute(G, attr):
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, attr).items())],dtype="float64")

def remove_edge(G, u, v):
    G.remove_edge(u,v)
    G = _label_edges_with_id(G)
    return G

def update_edge_attribute(G, attr, vector):
    d = dict(zip(sorted(G.edges()), vector))
    nx.set_edge_attributes(G, d, attr)
    return G