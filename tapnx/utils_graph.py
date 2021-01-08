"""Graph utility functions."""

import networkx as nx

def graph_from_edgedf(df_edges, edge_attr=None):
    """
    Convert node and edge DataFrames to a MultiDiGraph.
    Parameters
    ----------
    df_edges : Pandas dataframe
        DataFrame of edges
    df_node : Pandas dataframe 
        DataFrame of node positions
    edge_attr : Bool
        Include attributes in graph
    Returns
    -------
    G : networkx.DiGraph
    """
    # create a new networkx graph based on the edges dataframe edges_df
    G = nx.from_pandas_edgelist(df_edges, source='source', target='target',
                                 edge_attr=edge_attr, create_using=nx.DiGraph())
     
    return G

def graph_positions_from_nodedf(G, df_nodes):
    # read and format the positional data
    pos = df_nodes.to_dict(orient='index')
    G.graph['pos'] = {id:(pos[id][x],pos[id][y]) for id, (x,y) in pos.items()}

    return G