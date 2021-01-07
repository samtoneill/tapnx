"""Helper functions"""


def TNTP_to_pandas(net_file, node_file, trips_file, flow_file=None):

    """
    Converts a set of TNTP files to panda dataframes.

    https://github.com/bstabler/TransportationNetworks

    Parameters
    ----------
    file : TNTP file {}_net.tntp
        TNTP net file
    file : TNTP file {}_node.tntp
        TNTP node file
    file : TNTP file {}_trips.tntp
        TNTP trips file
    file : TNTP file {}_flow.tntp
        Optional - TNTP flow file
    Returns
    -------
    pandas dataframe: 
        pandas dataframe containing networkx edgelist 
    pandas dataframe: 
        pandas dataframe containing node positional data (x,y) 
    pandas dataframe: 
        pandas dataframe containing origin/destination data
    pandas dataframe: 
        pandas dataframe containing the optimal solution 
    """
    print('Convert ...')

def TNTP_net_to_pandas(file):
    """
    Converts a TNTP net file to panda edge dataframe.

    https://github.com/bstabler/TransportationNetworks

    Parameters
    ----------
    file : TNTP file {}_net.tntp
        TNTP net file
    Returns
    -------
    pandas dataframe: 
        pandas dataframe containing networkx edgelist 
    """

    print('converting TNPT net file to pandas edge dataframe')

def TNTP_node_to_pandas(file):
    print('converting TNPT node file to pandas node dataframe')

def TNTP_trips_to_pandas(file):
    print('converting TNPT trips file to pandas dataframe')

def TNTP_flow_to_pandas(file):
    print('converting TNPT flow file to pandas dataframe')

