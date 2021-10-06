import pandas as pd
import tapnx as tapnx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time as time

filename = 'MATLAB_test'



filename = 'siouxfallswithspeeds'
#filename = 'smallsiouxfalls'
#filename = 'siouxfalls'

nodes = None

G = tapnx.graph_from_csv(filename, nodes=nodes, trips=True, edge_attr=True)
if nodes:
    fig, ax = tapnx.plot_graph(G, node_size=200, node_labels=True)
    plt.show()

G.graph['no_edges'] = G.number_of_edges()

remote = True
otol=1.0e-3
f,cf,x,cx,min_d,_,_ = tapnx.gekko_optimise_column_gen(G,lam=1, remote=remote,d=True,otol=otol)
f,cf,x,cx,max_d,_,_ = tapnx.gekko_optimise_column_gen(G,min_max_type=-1, lam=1, remote=remote,d=True,otol=otol)

f,cf,x,cx,min_tt,_,_ = tapnx.gekko_optimise_column_gen(G,lam=0, remote=remote,d=True)
f,cf,x,cx,max_tt,_,_ = tapnx.gekko_optimise_column_gen(G,min_max_type=-1, lam=0, remote=remote,d=True,otol=otol)

max_tt = -max_tt
max_d = -max_d



dists = []
tts = []
lams = np.round(np.arange(0,1.01,0.2),2)
#lams = np.round(np.arange(0.15,0.3,0.01),2)
#lams = np.arange(0,1.01,0.1)
for lam in lams:
    #print(lam)
    #time.sleep(1)
    f,cf,x,cx,min_ws,dist,tt = tapnx.gekko_optimise_column_gen(G,lam=lam,remote=remote,min_d=min_d,max_d=max_d,min_tt=min_tt,max_tt=max_tt,d=True,otol=otol)
    dists.append(dist)
    tts.append(tt)


print(min_d)
print(max_d)
print(dists)
print(min_tt)
print(max_tt)
print(tts)
print((np.array(dists)-min_d)/(max_d-min_d))
print((np.array(tts)-min_tt)/(max_tt-min_tt))
plt.figure()
plt.plot(dists,tts, 'o')

plt.figure()
plt.plot(lams,dists)

plt.figure()
plt.plot(lams,tts)

plt.show()