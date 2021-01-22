"""Helper functions"""

import numpy as np
import os
import pandas as pd



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

def TNTP_net_to_pandas(filename):
    """
    Converts a TNTP net file to panda edge dataframe.

    https://github.com/bstabler/TransportationNetworks

    Parameters
    ----------
    filename : filename (with location)
        TNTP net file
    Returns
    -------
    pandas dataframe: 
        pandas dataframe containing networkx edgelist 
    """

    print('converting TNPT net file to pandas edge dataframe')

    df_net = pd.read_csv(filename, header=6, sep='\t')
    # clean up dataframe
    df_net.columns = df_net.columns.str.strip()
    df_net.drop(['~', ';'], axis=1, inplace=True)
    
    s = [int(item) for item in df_net['init_node'].tolist()]
    t = [int(item) for item in df_net['term_node'].tolist()]
    a = np.array([item for item in df_net['free_flow_time'].tolist()], dtype=np.float64)
    b = np.array(df_net['b'].tolist(), dtype=np.float64)
    c = np.array(df_net['capacity'].tolist(), dtype=np.float64)
    n = np.array(df_net['power'].tolist(), dtype=np.float64)

    data = list(zip(s,t,a,b,c,n))
    df_edges = pd.DataFrame(data, columns =['source', 'target', 'a', 'b', 'c', 'n']) 
    df_edges.to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
    print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))



def TNTP_node_to_pandas(filename):
    print('converting TNPT node file to pandas node dataframe')
    df_nodes = pd.read_csv(filename, sep='\t')
    # clean up dataframe
    df_nodes.columns = df_nodes.columns.str.strip()
    df_nodes.drop([';'], axis=1, inplace=True)

    df_nodes = pd.DataFrame.from_dict(dict(zip(df_nodes['Node'], zip(df_nodes['X'], df_nodes['Y']))), orient='index', columns =['X', 'Y'])
    # note that we want the node index to start at 0
    df_nodes.to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
    print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))


def TNTP_trips_to_pandas(filename):
    print('converting TNPT trips file to pandas dataframe')
    df = pd.DataFrame()
    commodities_dict = {}
    with open(filename) as in_file:
      data = []
      data_bool = False
      for index, line in enumerate(in_file):
        #print(index)
        #print(line)
        if index > 4:
          if line[0]=='O':
            if data_bool and data_row:
              commodities_dict[source_node] = data_row
              data.append(data_row)
            data_bool = True
            data_row = {}
            source_node = int(line.replace("Origin ", ""))
          else:
            
            line = line.strip()[:-1]
            #print(line)
            col_demand = [l.strip().split(':') for l in line.split(";")]
            if len(col_demand) > 1:
              #print(col_demand)
              col_demand = [[c.strip() for c in l] for l in col_demand]
              #print(col_demand) 
              for cd in col_demand:
                data_row[cd[0]] = cd[1]

    if data_bool:
      data.append(data_row)  
      commodities_dict[source_node] = data_row

    pd.DataFrame.from_dict(commodities_dict, orient='index').to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
    print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))

def TNTP_flow_to_pandas(file):
    print('converting TNPT flow file to pandas dataframe')

