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

# ## system optimal
# def t(x,a,b,c,n,d,lam):
#   return a*(1+b*(x/c)**n)

# def dtdx(x,a,b,c,n,d,lam):
#   return a*b*n*(c**(-n))*(x**(n-1))

# def d2tdx2(x,a,b,c,n,d,lam):
#   return  (n-1)*a*b*n*(c**(-n))*(x**(n-2))

# def edge_func_so_np(x,a,b,c,n,d,lam):
#   return t(x,a,b,c,n,d,lam) + x*dtdx(x,a,b,c,n,d,lam)

# def edge_func_so_derivative_np(x,a,b,c,n,d,lam):
#   return 2*dtdx(x,a,b,c,n,d,lam) + x*d2tdx2(x,a,b,c,n,d,lam)

## distance system optimal

##
# need F = lam*D + lam*T
# D = d*x
# T = x*t(x)

# to solve via equilibrium use 
# t_hat = F + x*F'
# F' = lam*D' + lam*T'
#    = lam*d + lam*( t(x) + x*t'(x) )

# set max_d and max_tt to 1 for standard scale
# max_tt should be divided by the number of edges as the total travel time is a sum of all edges
# to maximise travel time, minimise -t, so lam = 0, max_tt =-1, results in a negative 1 
def t(x,a,b,c,n,d,lam,max_d,max_tt):
  return (lam/max_d)*d + ((1-lam)/max_tt)*a*(1+b*(x/c)**n)

def dtdx(x,a,b,c,n,d,lam,max_d,max_tt):
  return ((1-lam)/max_tt)*a*b*n*(c**(-n))*(x**(n-1))

def d2tdx2(x,a,b,c,n,d,lam,max_d,max_tt):
  return ((1-lam)/max_tt)*(n-1)*a*b*n*(c**(-n))*(x**(n-2))

def edge_func_dist_np(x,a,b,c,n,d,lam,max_d,max_tt):
  return t(x,a,b,c,n,d,lam,max_d,max_tt) + x*dtdx(x,a,b,c,n,d,lam,max_d,max_tt)

def edge_func_dist_derivative_np(x,a,b,c,n,d,lam,max_d,max_tt):
  return 2*dtdx(x,a,b,c,n,d,lam,max_d,max_tt) + x*d2tdx2(x,a,b,c,n,d,lam,max_d,max_tt)


filename = 'MATLAB_test'

# meta_data = tapnx.readTNTPMetadata('test_data/{}/{}_net.tntp'.format(filename,filename))
# df_edges = tapnx.TNTP_net_to_pandas('test_data/{}/{}_net.TNTP'.format(filename, filename), start_line=meta_data['END OF METADATA'])

#df_nodes = tapnx.TNTP_node_to_pandas('test_data/{}/{}_node.TNTP'.format(filename, filename))
# df_trips = tapnx.TNTP_trips_to_pandas('test_data/{}/{}_trips.TNTP'.format(filename, filename))

df_edges, df_nodes, df_trips = tapnx.graph_from_csv(
    edges_filename = 'test_data/{}/{}_net.csv'.format(filename, filename),
    trips_filename = 'test_data/{}/{}_trips.csv'.format(filename, filename)
)
G = tapnx.graph_from_edgedf(df_edges, edge_attr=True)
G = tapnx.trips_from_tripsdf(G, df_trips)

n = tapnx.get_np_array_from_edge_attribute(G, 'n')
b = tapnx.get_np_array_from_edge_attribute(G, 'b')
d = tapnx.get_np_array_from_edge_attribute(G, 'd')
#m = tapnx.get_np_array_from_edge_attribute(G, 'm')
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


tol = 10**-6
max_iter = 100
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


lam = 0
max_tt = 1
max_d = 1
G, data = tapnx.gradient_projection(
    G,
    collect_data=True,
    aec_gap_tol=tol,
    max_iter=max_iter,
    d=False,
    lam=lam)

x = data['x'][-1]
#print(x)
#print(t(x,a,b,c,n,d,lam,max_d,max_tt))
UE = np.sum(x*t(x,a,b,c,n,d,lam,max_d,max_tt))

lam = 0
max_tt = 1
max_d = 1
G, data = tapnx.gradient_projection(
    G,
    collect_data=True,
    aec_gap_tol=tol,
    max_iter=max_iter,
    edge_func=edge_func_dist_np,
    edge_func_derivative= edge_func_dist_derivative_np,
    d=False,
    lam=lam)

x = data['x'][-1]
#print(x)
#print(t(x,a,b,c,n,d,lam,max_d,max_tt))
SO = np.sum(x*t(x,a,b,c,n,d,lam,max_d,max_tt))

lam = 1
max_tt = 1
max_d = 1
G, data = tapnx.gradient_projection(
    G,
    collect_data=True,
    aec_gap_tol=tol,
    max_iter=max_iter,
    edge_func=edge_func_dist_np,
    edge_func_derivative= edge_func_dist_derivative_np,
    d=True,
    lam=lam)

x = data['x'][-1]
print(x)
#print(t(x,a,b,c,n,d,lam,max_d,max_tt))
WEI_SO = np.sum(x*t(x,a,b,c,n,d,lam,max_d,max_tt))

print(SO)
print(WEI_SO)
print(UE)
print(UE/SO)

# get max by maximising total travel time, min -xt(x)


# get max distance by maximising distance, min -dx




# tt_results = []
# d_results = []

# # lam up to 0.9999
# for lam in np.arange(0,1,0.01):
#   print(lam)
#   G, data = tapnx.gradient_projection(
#     G,
#     collect_data=True,
#     aec_gap_tol=tol,
#     max_iter=max_iter,
#     edge_func=edge_func_dist_np,
#     edge_func_derivative= edge_func_dist_derivative_np,
#     d=True,
#     lam=lam)

#   x = data['x'][-1]
#   #print(x)
#   #print(t(x,a,b,c,n,d,lam))

#   tt_results.append(np.sum(x*t(x,a,b,c,n,d,0)))
#   d_results.append(np.sum(x*t(x,a,b,c,n,d,1)))
  
# print(tt_results)
# print(d_results)
# plt.figure()
# plt.plot(d_results, tt_results, 'o')
# plt.show()

  #plt.figure()
  #plt.plot(data['AEC'], label='Gradient Projection 1')
  #plt.yscale('log')

