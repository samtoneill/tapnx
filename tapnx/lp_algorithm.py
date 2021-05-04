""" Column generation with piecewise linear approximation objective for TAP """
# https://www.gurobi.com/documentation/9.0/examples/piecewise_py.html

import gurobipy as gp 
from gurobipy import GRB
from collections import defaultdict
import numpy as np
import networkx as nx
from . import utils_graph
import numpy as np
import time as time

def milp_tap(G):
    """
    Solve the TAP problem with MILP
    """

    # dictionary to store data about the instance the method was run
    data = {'AEC':[], 'relative_gap':[], 'x':[], 'weight':[], 'objective':[]}

    no_edges = G.graph['no_edges']

    # edge attributes used in the calculation of edge travel times
    a = utils_graph.get_np_array_from_edge_attribute(G, 'a')
    b = utils_graph.get_np_array_from_edge_attribute(G, 'b')
    cap = utils_graph.get_np_array_from_edge_attribute(G, 'c')
    n = utils_graph.get_np_array_from_edge_attribute(G, 'n')

    # initialise edge flow x and edge travel time t
    x = np.zeros(no_edges,dtype="float64") 
    t = _edge_func_np(x,a,b,cap,n)

    trips = G.graph['trips']

    #dictionary to store shortest paths, indexed by (origin, destination)
    shortest_paths = dict()

    # dictionary to store paths for an (origin, destination)
    # e.g. {(origin, destination):{{'paths':[1,2,3] 'cost', 4, 'flow',1}} stores a path [1,2,3] with length 4 and flow 1
    od_paths = defaultdict(lambda: {})
    edge_paths_indexes = defaultdict(lambda: [])
    # also need edges given by path edges

    edges = {G[u][v]['id']:(u,v) for u,v in sorted(G.edges())}
    edge_ids = edges.keys()

    demand_od = {}
    od_indexes = {}
    indx = 0
    for origin, destinations in trips.items():
        for destination in destinations:
            demand = trips[origin][destination]
            if not ((demand == 0) or (demand == np.nan)):
                od_indexes[(int(origin), int(destination))] = indx
                indx += 1


    m = gp.Model('tap_lp')
    # for LP we need pi_j and pi sets. These are defined by od_paths
    

    
    # also need path flow for commodity (i,j) and path index k
    # path_flow[i,j,k]

    xie = np.linspace(0,300600,1000)
    len_i = len(xie)
    xi = np.zeros((no_edges, len_i))
    yi = np.zeros((no_edges, len_i))

    # update to numpy vector operations
    for e in edge_ids:
        xi[e,:] = xie 
        yi[e,:] = objective(xi[e,:],a[e],b[e],cap[e],n[e])


    lam_indexes = [k for k in range(len(xie)-1)]


    # define the variables
    # flow on link e
    x_e = m.addVars(edge_ids, name="x_e")
    # cost of flow on link e
    y_e = m.addVars(edge_ids, name="y_e")

    # lambda variable for link e at piecewise index k
    lam_ei = m.addVars(edge_ids, lam_indexes, name="lam_e")

    # delta binary for link e
    delta_e = m.addVars(edge_ids, vtype=GRB.BINARY, name="delta_e")

    m.addConstrs(
        (gp.quicksum(lam_ei[e,i]*xi[e,i] for i in lam_indexes) == x_e[e] for e in edge_ids), 
        name='lambda_constraints_x'
    )

    m.addConstrs(
        (gp.quicksum(lam_ei[e,i]*yi[e,i] for i in lam_indexes) == y_e[e] for e in edge_ids), 
        name='lambda_constraints_y'
    )

    m.addConstrs(
        (gp.quicksum(lam_ei[e,i] for i in lam_indexes) == 1*delta_e[e] for e in edge_ids), 
        name='lambda_constraints_1'
    )

    
    i = 0
    while True:
        print('Iteration i = {}--------------------\n\n\n'.format(i))
        
        # store the flow and weight at the start of the iteration (used for AEC)
        x_aec = np.copy(x)
        t_aec = np.copy(t)
        utils_graph.update_edge_attribute(G, 'weight',t)

        no_new_paths = 0
        for origin, destinations in trips.items():
            # Create a copy of the graph and remove the out edges that are connected to zones based on whether they are 
            # prohibited to be thru nodes

            J = G.copy()
            if G.graph['first_thru_node'] > 1:
                print('here')
                print(G.graph['first_thru_node'] )
                zones_prohibited = list(range(1,origin))
                zones_prohibited += list(range(origin+1, G.graph['first_thru_node']))
                out_edges_for_removal = G.out_edges(zones_prohibited)
                J.remove_edges_from(out_edges_for_removal)
                #print(zones_prohibited)

            # calculate shortest paths, networkx does not (check) have an option for a list of targets
            lengths, all_paths = nx.single_source_dijkstra(J, source=origin, target=None, weight='weight')
            
            # iterate through the destinations for an origin
            for destination in destinations:
                # calculate shortest paths, networkx does not (check) have an option for a list of targets
                
                # get demand associated with origin/destination
                demand = trips[origin][destination]
                
                # if there are positive trips (otherwise no need to do anything)
                if not ((demand == 0) or (demand == np.nan)):
                    od_index = od_indexes[(int(origin), int(destination))]
                    # get the shortest path and its length for the origin/destination
                    shortest_path = all_paths[int(destination)]
                    shortest_path_length = lengths[int(destination)]
                    # get the paths for the 
                    paths = list(od_paths[od_index].values())
                    path_edge_ids = [G[u][v]['id'] for u,v in utils_graph.edges_from_path(shortest_path)]

                    # overwrite the shortest path
                    #shortest_paths[(origin,destination)] = {'path': shortest_path, 'path_length': shortest_path_length}

                    # if the shortest path is not yet in the paths, add it
                    if not path_edge_ids in paths:
                        no_new_paths += 1
                        for e in edge_ids:
                            if e in path_edge_ids:
                                edge_paths_indexes[e].append( (od_index, len(paths))) 
                    
                        od_paths[od_index][len(paths)] = path_edge_ids

        if no_new_paths == 0:
            break
                
        
        # lambda constraints



        path_indexes = {}
        for od_index, paths in od_paths.items():
            path_indexes[od_index] = [int(j) for j, path in paths.items()]

          
        od_path_indexes = [(od_index,j) for od_index, paths in od_paths.items() for j, path in paths.items()]

        od_ids, demand = gp.multidict(
            {od_index: trips[o][d] for (o,d), od_index in od_indexes.items()}
        )


        # flow on path j for commodity k
        f_kj = m.addVars(od_path_indexes, name="f_kj")

        # # definitional sum of flows on an edge = flow on edges

        ## THIS IS WRONG, NEEDS TO BE SETS THAT CONTAIN PATHS THAT INCLUDE EDGE e

        def_con = m.addConstrs(
            (gp.quicksum(f_kj[k,j] for (k,j) in edge_paths_indexes[e]) == x_e[e] for e in edge_ids), 
            name='def_constraints'
        )

        # # demand sum of flows for (o,d) = demand for (o,d)
        
        dem_con = m.addConstrs(
            (gp.quicksum(f_kj[k,j] for j in path_indexes[k]) == demand[k] for k in od_ids), 
            name='demand_constraints'
        )

        # # sum of all deltas: Note that as we don't have all paths this might be infeasible.
        # solution is to generate 3-5 paths for each OD to start with.
        # removal of two many links obviously affects the column generation and the feasibility of the model
        # if i>10:

        #     m.addConstr(
        #         (gp.quicksum(delta_e[e] for e in edge_ids) <= 75), 
        #         name='delta_constraint'
        #     )

        m.setObjective(gp.quicksum(y_e[e] for e in edge_ids))
        #m.setObjective(gp.quicksum(delta_e[e] for e in edge_ids) )
        #m.setObjective(gp.quicksum(y_e[e] for e in edge_ids) + gp.quicksum(delta_e[e] for e in edge_ids)*10**5 )
        m.update()
        m.write("model.lp")

        m.optimize()

        # print('IsMIP: %d' % m.IsMIP)
        # for v in m.getVars():
        #     print('%s %g' % (v.VarName, v.X))
        # print('Obj: %g' % m.ObjVal)
        # print('')
        deltas = []
        for e in edge_ids:
            x[e] = float(m.getVarByName("x_e[{}]".format(e)).x)
            deltas.append(float(m.getVarByName("delta_e[{}]".format(e)).x))
            if deltas[-1] < 0.001:
                print(e)
                print(cap[e])
                print(edges[e])
                for od_index, path_index in edge_paths_indexes[e]:
                    od_paths[od_index][path_index]

        print(x)
        print(deltas)
        t = _edge_func_np(x,a,b,cap,n)                        

        i += 1   

        m.remove(def_con)
        m.remove(dem_con)
        m.remove(f_kj)

        
    return True


def initialise_model():
    """
    Initialise MILP model
    """

    demand = np.transpose(np.array([[3,2]]))
    lb = 0.0
    ub = np.sum(demand)
    no_edges = 5
    no_paths = 4
    is_integer = False
    edges = range(no_edges)

    # np.random.seed(1)
    # a = np.random.random(no_edges)*5
    # c = np.random.random(no_edges)*10 +1

    a = np.transpose(np.array([1, 2, 2, 1, 1]))
    b = np.transpose(np.array([0, 0, 1/2, 2, 0]))
    c = np.transpose(np.array([1, 1, 1, 1, 1]))
    n = np.transpose(np.array([1, 1, 1, 1, 1]))

    D = np.array([[1,1,0,0],
                [0,0,1,1],
                [1,0,1,0],
                [0,1,0,1],
                [1,1,1,1]])

    A = np.array([[1,1,0,0], 
                [0,0,1,1]])

    return True

def update_model_paths():
    """
    Update the current MILP model with new link-path matrix
    """
    return True

def get_D_matrix(od_paths):
    np_arrays = tuple([value['D'] for key, value in od_paths.items()])
    #flatten list
    np_arrays = [item for sublist in np_arrays for item in sublist]
    print(np_arrays)
    print(np.column_stack(np_arrays))


def _edge_func_np(x,a,b,c,n):
    return a*(1+b*(x/c)*(x/c)*(x/c)*(x/c))

def _edge_func_derivative_np(x,a,b,c,n):
    return (a*b*n*x*x*x)/(c*c*c*c)

def objective(x,a,b,c,n):
    return a*x*(1 + (b/(n+1))*(x/c)**n) 

