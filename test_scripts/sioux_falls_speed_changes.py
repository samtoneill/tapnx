import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def travel_time(x,c,d,m):
  a = d/m
  #b = a*0.15

  return a*(1+0.15*(x/c)**4)

def total_travel_time(x,c,d,m):
  return np.sum(x*travel_time(x,c,d,m))

def speed(x,c,d,m):  
  a = d/m
  #b = a*0.15
  
  return m/(1+0.15*(x/c)**4)

def fuel_consumption(x,c,d,m):
  s = speed(x,c,d,m)
  return 0.0019*s**2 -0.2784*s + 17.337;

def total_fuel_consumption(x,c,d,m):
  return np.sum((d/100)*x*fuel_consumption(x,c,d,m))


filename = 'siouxfallswithspeeds'

# meta_data = tapnx.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
# df_edges = tapnx.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])

#df_nodes = tapnx.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
# df_trips = tapnx.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    nodes_filename = 'test_data/{}/{}_node.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)
G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.trips_from_tripsdf(G, df_trips)

n = tapnx.get_np_array_from_edge_attribute(G, 'n')
b = tapnx.get_np_array_from_edge_attribute(G, 'b')
d = tapnx.get_np_array_from_edge_attribute(G, 'd')
m = tapnx.get_np_array_from_edge_attribute(G, 'm')
c = tapnx.get_np_array_from_edge_attribute(G, 'c')
a = tapnx.get_np_array_from_edge_attribute(G, 'a')
#a = d/m
#print(a)
G.graph['no_edges'] = len(a)
G.graph['first_thru_node'] = 0
# G.graph['name'] = filename
# G.graph['no_zones'] = int(meta_data['NUMBER OF ZONES'])
# G.graph['no_nodes'] = int(meta_data['NUMBER OF NODES'])
# G.graph['first_thru_node'] = int(meta_data['FIRST THRU NODE'])
# G.graph['no_edges'] = int(meta_data['NUMBER OF LINKS'])
#G = tapnx.graph_positions_from_nodedf(G, df_nodes)


tol = 10**-3
max_iter = 50
#G, data = tapnx.gradient_projection(G,collect_data=True,aec_gap_tol=tol,max_iter=max_iter)
# plt.plot(data['AEC'], label='Gradient Projection 1')
# plt.plot(data['no_paths'], label='No. paths')
# #print(data_fw['x'][-1])
#print(data['x'][-1])
#print(np.sum(data['objective'][-1]))
# lam = 0
# G, data = tapnx.gradient_projection(
#   G,
#   collect_data=True,
#   aec_gap_tol=tol,
#   max_iter=max_iter)

# x = data['x'][-1]
#print(x)
#print(t(x,a,b,c,n,d,lam))
# print(np.sum(x*t(x,a,b,c,n,d,lam)))
#plt.figure()
#plt.plot(data['AEC'], label='Gradient Projection 1')
#plt.yscale('log')


  #plt.figure()
  #plt.plot(data['AEC'], label='Gradient Projection 1')
  #plt.yscale('log')


# tfc_results = []
# tt_results = []

# for i in range(len(a)):
#     print(i)
#     m_new = np.copy(m)
#     m_new[i]+=5
#     a_new = d/(m_new/60)
#     tapnx.update_edge_attribute(G, 'a', a_new)

#     G, data = tapnx.gradient_projection(
#       G,
#       collect_data=True,
#       aec_gap_tol=tol,
#       max_iter=max_iter)
#     x = data['x'][-1]
#     tfc = total_fuel_consumption(x,c,d,m)
#     tfc_results.append(tfc)
#     tt = total_travel_time(x,c,d,m)
#     tt_results.append(tt)
#     print(np.sum(data['objective'][-1]))

# plt.plot(tfc_results, tt_results, 'o')
# plt.xlabel('No. Iterations')
# plt.ylabel('AEC')
# plt.yscale('log')
# plt.legend()

# plt.figure()
# plt.plot(data['no_paths'], data['AEC'])
# plt.yscale('log')

# plt.show()
