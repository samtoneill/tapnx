"""Graph utility functions."""

import networkx as nx
import numpy as np
import pandas as pd
import time
import copy

from . import helper

def edges_from_path(path):
    """
    Converts the list of path nodes into a list of path edges 
    ----------
    path: list of path nodes
    Returns
    -------
    path of edge: list of path edges
    """
    return list(zip(path,path[1:]))

def path_length(G, path):
    """
    Get the length of a given path from a list of path nodes
    Parameters
    ----------
    path: list
    list of path nodes
    Returns
    -------
    path length: double
    """
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
    #G.graph['no_edges'] = G.number_of_edges()
    G.graph['no_nodes_in_original'] = G.number_of_nodes()
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

def graph_from_csv(filename, nodes=None, trips=None, edge_attr=None):

    
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename)
    nodes_filename = None
    trips_filename = None
    if nodes:
        nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename)
    if trips:
        trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)

    df_edges, df_nodes, df_trips = graph_data_from_csv(
        edges_filename = edges_filename,
        nodes_filename = nodes_filename,
        trips_filename = trips_filename
    ) 
    G = graph_from_edgedf(df_edges, edge_attr=edge_attr)
    if nodes:
        G = graph_positions_from_nodedf(G, df_nodes)
    if trips:
        G = trips_from_tripsdf(G, df_trips)
    return G

def graph_data_from_csv(edges_filename, nodes_filename=None, trips_filename=None):
    """
    Construct a set of CSV files to dataframes for graph processing
    ----------
    edges_filename: string
        Filename of csv file constaing edge information
    nodes_filename: string
        Filename of csv file constaing node information
    trips_filename: string
        Filename of csv file constaing edge information
    Returns
    -------
    G : networkx.DiGraph
        Networkx directed graph
    """

    df_edges = pd.read_csv(edges_filename, index_col=[0])
    df_nodes = None
    df_trips = None

    if nodes_filename:
        df_nodes = pd.read_csv(nodes_filename, index_col=[0])
    
    if trips_filename:
        df_trips = pd.read_csv(trips_filename, index_col=0)
        df_trips.fillna(0, inplace=True)

    return df_edges, df_nodes, df_trips

def _label_edges_with_id(G):
    """
    Give a unique integer label to the graph edges
    ----------
    G : networkx.DiGraph
        Networkx directed graph
    Returns
    -------
    G : networkx.DiGraph
        Networkx directed graph
    """
    for index, (u,v) in enumerate(sorted(G.edges(), key= lambda edge: (edge[0], edge[1]))):
        G[u][v]['id'] = index
    return G

def get_np_array_from_edge_attribute(G, attr):
    """
    Generic method that returns a numpy array of the specified attribure for the sorted edges
    ----------
    G : networkx.DiGraph
        Networkx directed graph
    attr: string
          The name of an edge attribure
    Returns
    -------
    1D numpy array
        array containg the specified atrribute of all edges
    """
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, attr).items())],dtype="float64")


def _remove_edge(G, u, v):
    H = copy.deepcopy(G)
    H.remove_edge(u,v)
    H = _label_edges_with_id(H)
    return H

def _remove_node(G, n):
    H = copy.deepcopy(G)
    H.remove_node(n)
    H = _label_edges_with_id(H)
    return H

def remove_edge(G,u,v):
    # remove edge
    # Find all OD pairs that have zero shortest paths
    # for each OD pair
        # remove trips from trip table
    G = _remove_edge(G,u,v)
    return G

def remove_node(G,n,remove_trips=False):
    # remove node
    # Find all OD pairs that have zero shortest paths
    # for each OD pair
        # remove trips from trip table
    G = _remove_node(G,n)

    # we can remove the trips from the table
    if remove_trips:
        trips = G.graph['trips']
        # delete trips from origin
        del trips[n]
        # delete trips to origin
        for key, value in trips.items():
            del value['{}'.format(n)]

        G.graph['trips'] = trips

    return G

def update_edge_attribute(G, attr, vector):
    d = dict(zip(sorted(G.edges()), vector))
    nx.set_edge_attributes(G, d, attr)
    return G

def graph_from_TNTP(filename, nodes=None, edge_attr=None):
    
    meta_data = helper.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
    df_edges = helper.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])
    df_trips = helper.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))

    G = graph_from_edgedf(df_edges, edge_attr=True)
    G = trips_from_tripsdf(G, df_trips)

    G.graph['name'] = filename
    G.graph['no_zones'] = int(meta_data['NUMBER OF ZONES'])
    G.graph['no_nodes'] = int(meta_data['NUMBER OF NODES'])
    G.graph['first_thru_node'] = int(meta_data['FIRST THRU NODE'])
    G.graph['no_edges'] = int(meta_data['NUMBER OF LINKS'])

    if nodes:
        df_nodes = helper.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
        G = graph_positions_from_nodedf(G, df_nodes)


    return G
