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

def TNTP_net_to_pandas(filename, start_line, save=False):
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

    df_net = pd.read_csv(filename, header=start_line, sep='\t')
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
    if save:
        df_edges.to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
        print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))
    return df_edges


def TNTP_node_to_pandas(filename, save=False):
    print('converting TNPT node file to pandas node dataframe')
    df_nodes = pd.read_csv(filename, sep='\t')
    # clean up dataframe
    df_nodes.columns = df_nodes.columns.str.strip()
    df_nodes.drop([';'], axis=1, inplace=True)

    df_nodes = pd.DataFrame.from_dict(dict(zip(df_nodes['node'], zip(df_nodes['X'], df_nodes['Y']))), orient='index', columns =['X', 'Y'])
    # note that we want the node index to start at 0
    if save:
        df_nodes.to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
        print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))

    return df_nodes


def TNTP_trips_to_pandas(filename, save=False):
    print('converting TNPT trips file to pandas dataframe')
    #df = pd.DataFrame()
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
              
              col_demand = [[float(c.strip()) for c in l] for l in col_demand]

              for cd in col_demand:
                data_row[cd[0]] = cd[1]

    if data_bool:
      data.append(data_row)  
      commodities_dict[source_node] = data_row

    df_trips = pd.DataFrame.from_dict(commodities_dict, orient='index')
    df_trips.fillna(0, inplace=True)
    if save:
        df_trips.to_csv('{}.csv'.format(os.path.splitext(filename)[0]))
        print('Saved file to {}'.format('{}.csv'.format(os.path.splitext(filename)[0])))
    return df_trips

def TNTP_flow_to_pandas(file):
    print('converting TNPT flow file to pandas dataframe')


def readTNTPMetadata(demand_filename):
    """
    Read metadata tags and values from a TNTP file, returning a dictionary whose
    keys are the tags (strings between the <> characters) and corresponding values.
    The last metadata line (reading <END OF METADATA>) is stored with a value giving
    the line number this tag was found in.  You can use this to proceed with reading
    the rest of the file after the metadata.
    """
   
    with open(demand_filename, "r") as demand_file:
       lines = demand_file.read().splitlines()

    metadata = dict()
    lineNumber = 0
    for line in lines:
        lineNumber += 1
        line.strip()
        commentPos = line.find("~")
        if commentPos >= 0: # strip comments
            line = line[:commentPos]
        if len(line) == 0:
            continue

        startTagPos = line.find("<")
        endTagPos = line.find(">")
        if startTagPos < 0 or endTagPos < 0 or startTagPos >= endTagPos:
            print("Error reading this metadata line, ignoring: '%s'" % line)
        metadataTag = line[startTagPos+1 : endTagPos]
        metadataValue = line[endTagPos+1:]
        if metadataTag == 'END OF METADATA':
            metadata['END OF METADATA'] = lineNumber
            return metadata
        metadata[metadataTag] = metadataValue.strip()
      
    print("Warning: END OF METADATA not found in file")
    return metadata

# no zones == no_nodes. e.g. sioux falls 
def pandas_trips_to_TNTP(df_trips, tntp_filename, no_zones):
    df_trips.fillna(0, inplace=True)
    
    with open(tntp_filename, 'w') as f:
        f.write("<NUMBER OF ZONES> {}\n".format(no_zones))
        f.write("<TOTAL OD FLOW> {}\n".format(df_trips.sum().sum()))
        f.write("<END OF METADATA>\n\n\n".format(no_zones))
        for key, value in df_trips.iteritems():
            f.write("Origin  {}\n".format(key))
            l_all = ["\t{} :\t{};".format(index,v) for index, v in value.iteritems()]
            for l in [l_all[x:x+5] for x in range(0, len(l_all),5)]:
                f.write("".join(l))
                f.write("\n")
            f.write("\n\n")       
    return None

# this is only applicable for speed =0 and toll= 0 and length = free flow time. 
# also number of zones = number of nodes
# see Sioux Falls
def pandas_net_to_TNTP(df_edges, tntp_filename, no_nodes, no_edges):
    with open(tntp_filename, 'w') as f:
        f.write("".join(["<NUMBER OF ZONES>", " {}".format(no_nodes),"\n\n"]))
        f.write("".join(["<NUMBER OF NODES>", " {}".format(no_nodes),"\n\n"])) 
        f.write("".join(["<FIRST THRU NODE>", " {}".format(1),"\n\n"]))
        f.write("".join(["<NUMBER OF LINKS>", " {}".format(no_edges),"\n"]))
        f.write("".join([
            "<ORIGINAL HEADER>~\t",
            "Init node\t",
            "Term node\t",
            "Capacity\t",
            "Length\t",
            "Free Flow Time\t",
            "B\t",
            "Power\t",
            "Speed\t",
            "limit\t",
            "Toll\t",
            "Type\t",
            ";\n",
            "<END OF METADATA>\n"
        ]))
        f.write("\n\n\n")
        f.write("".join(
            [
            "~\t",
            "init_node\t",
            "term_node\t",
            "capacity\t",
            "length\t",
            "free_flow_time\t",	
            "b\t",
            "power\t",
            "speed\t"
            "toll\t",
            "link_type\t",
            ";\n"
            ]
        ))
        for index,row in df_edges.iterrows():
            f.write("".join(["\t",str(int(row['source'])), 
                        "\t", str(int(row['target'])),
                        "\t", str(row['c']),
                        "\t", str(row['a']),
                        "\t", str(row['a']),
                        "\t", str(row['b']),
                        "\t", str(row['n']),
                        "\t", str(0),
                        "\t", str(0),
                        "\t", str(1),
                        "\t", ";",
                        "\n"] 
                        ))
    #with open('file_to_write', 'w') as f:
    #f.write('file contents\n')
    return None