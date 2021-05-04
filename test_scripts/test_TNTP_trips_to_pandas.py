import tapnx as tapnx

#tapnx.TNTP_trips_to_pandas('../test_data/siouxfalls/SiouxFalls_trips.tntp')
#tapnx.TNTP_trips_to_pandas('../test_data/chicagosketch/ChicagoSketch_trips.tntp')
#tapnx.TNTP_net_to_pandas('../test_data/siouxfalls/SiouxFalls_net.tntp')
#tapnx.TNTP_node_to_pandas('../test_data/siouxfalls/SiouxFalls_node.tntp')
filepath = r'C:\Users\Sam\Documents\GitHub\TransportationNetworks'
filename = 'anaheim'
tapnx.TNTP_trips_to_pandas('{}/{}/{}_trips.tntp'.format(filepath,filename,filename))
tapnx.TNTP_net_to_pandas('{}/{}/{}_net.tntp'.format(filepath,filename,filename))
