def graph_from_edgelist(df_edges, df_nodes=None, edge_attr=False):
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
    # create a new networkx graph based on the edges datafram edges_df
    G = nx.from_pandas_edgelist(edges_df, source='source', target='target',
                                 edge_attr=edge_attr, create_using=nx.DiGraph())
    return G