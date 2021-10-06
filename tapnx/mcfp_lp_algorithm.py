import gurobipy as gp 
from gurobipy import GRB
from collections import defaultdict
import numpy as np
import networkx as nx
#from . import utils_graph
import numpy as np
import time as time
from functools import reduce

def mcfp_path_based():
    """
    Solve the MCFP via column generation and LP
    """
    
    # commodites and demands 
    # commodities = [(i, j, demand), ...]
    commodities = [(0,3,5), (2,5,3)]

    # demand 
    demand = [5,3]

    # edge upper capacity
    u_e = [None, None,5,2,None,None,None]

    # edge costs
    c_e = [1,5,1,1,6,1,1]

    # k_paths 
    # dictionary - key is commodity k, value is a list of paths, each path is a list of edges
    # {k:[[e,..],..]}
    k_paths = {
        0:[[1], [0,2,5]],
        1:[[4], [3,2,6]]
    }

    # for each commodity k and the associated paths j (note j index restarts for each k)
    # [(k,j),...]
    # used for generating flow vars
    commodity_path_pairs = [(0,0), (0,1),(1,0),(1,1)]

    # edge_ids 
    # [e, ...]
    edge_ids = [0,1,2,3,4,5,6]

    # commodity_path_ids - the paths available for commodity k
    # dictionary - key is commodity k, value is a list of path_ids j
    # {k:[j,]}
    commodity_path_ids = {
        0:[0,1],
        1:[0,1]
    }

    commodity_ids = commodity_path_ids.keys()

    # edge_path_ids - the paths that use edge e
    # dictionary - key is edge e, value is a list of path_ids (k,j) 
    # {e:[(k,j)]}
    # generate automatically via k_paths

    edge_path_ids = {e:reduce(lambda x, y: x+y,
                [list(
                    map(
                            lambda x: (k, x), 
                            [v.index(x) for x in filter(lambda x: bool(e in x), v)]
                        )
                    ) 
                    for k,v in k_paths.items()
                ]
            )
        for e in edge_ids
    }

    print(edge_path_ids)

    model_name = 'mcfp'
    m = mcfp_lp(
        commodity_path_pairs, 
        commodity_path_ids, 
        commodity_ids,
        edge_path_ids, 
        edge_ids,
        demand,
        c_e,
        u_e,
        model_name,
        no_of_paths=1
    )
    m.write('{}.lp'.format(model_name))
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))


def mcfp_lp(commodity_path_pairs, 
            commodity_path_ids, 
            commodity_ids,
            edge_path_ids, 
            edge_ids,
            demand,
            c_e,
            u_e,
            model_name,
            no_of_paths=None):
    
    m = gp.Model(model_name)
    M = 9999

    # flow on path j for commodity k
    f_kj = m.addVars(commodity_path_pairs, name="f_kj")
    x_e = m.addVars(edge_ids, name="x_e")
    
    # demand constraints for commodities
    dem_con = m.addConstrs(
        (gp.quicksum(f_kj[k,j] for j in commodity_path_ids[k]) == demand[k] for k in commodity_ids), 
        name='demand_constraints'
    )

    # edge flow is the sum of all path flows that use the edge
    def_con = m.addConstrs(
        (gp.quicksum(f_kj[k,j] for (k,j) in edge_path_ids[e]) == x_e[e] for e in edge_ids), 
        name='def_constraints'
    )

    # upper capacity limit of edge
    edge_capacity_con = m.addConstrs(
        (x_e[e] <= u_e[e] for e in edge_ids if u_e[e]),
        name='edge_capacity_constraints'
    )

    if no_of_paths:
        print('adding addtional constraint to limit the number of paths commodities can use')
        del_f_kj = m.addVars(commodity_path_pairs, name="del_f_kj",vtype=GRB.BINARY)
           # upper capacity limit of edge
        del_f_con_1 = m.addConstrs(
            (del_f_kj[k,j] <= M*f_kj[k,j] for k in commodity_ids for j in commodity_path_ids[k]),
            name='edge_capacity_constraints_1'
        )

        del_f_con_2 = m.addConstrs(
            (f_kj[k,j] <= M*del_f_kj[k,j] for k in commodity_ids for j in commodity_path_ids[k]),
            name='edge_capacity_constraints_2'
        )
        # number of paths j used for commodity k must be equal to the number of paths specified
        def_con = m.addConstrs(
            (gp.quicksum(del_f_kj[k,j] for j in commodity_path_ids[k]) == no_of_paths for k in commodity_ids), 
            name='edge_capacity_constraints_no_paths'
        )


    m.setObjective(gp.quicksum(c_e[e]*x_e[e] for e in edge_ids))
    
    m.update()
    return m


if __name__ == '__main__':
    mcfp_path_based()

    ''' TODO
        - Test orlin for simple MCFP
        - Need to automatically genereate all the ids etc from a csv graph or some data

    '''