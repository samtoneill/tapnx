import numpy as np

def algorithm_b():
    return True

def update_trees(k, n):
    """
    equalised path 
    
    Parameters
    ----------
    k : int
        exclusive lower bound on nodes to label
    n : int
        exclusive upper bound on nodes to label
    Returns
    -------
    del_c : float
        maximum difference of: max path cost - min path cost
    """
    return True

def shift_flow():
    return True

def get_branch_node():
    return True

def equalise_path_cost(k, j, x_min, x_max, c_min, c_max, c_prime_min, c_prime_max):
    """
    equalised path 
    
    Parameters
    ----------
    k : int
        start node of path to j
    j : int
        end node of path
    x_min : float
        total flow on minimum path
    x_max : float
        total flow on maximum path
    c_min : float 
        cost of minimum path
    c_max : float 
        cost of maximum path
    c_prime_min : float 
        derivative cost of minimum path
    c_prime_max : float 
        derivative cost of maximum path
    Returns
    -------
    
    """
    global epsilon
    global alpha_min
    global alpha_max

    # get the flow and cost differentials
    del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_prime_min, c_prime_max)
    
    # while the cost differential of a path segment is sufficiently large, shift flow
    while epsilon < del_c:
        x_min, c_min, c_prime_min = update_path_flow(del_x, k, j, alpha_min)
        x_max, c_max, c_prime_max = update_path_flow(del_x, k, j, alpha_max)
        # get the update flow and cost differentials
        del_x, del_c = get_delta_x_and_c(x_min, x_max, c_min, c_max, c_prime_min, c_prime_max)
        
    # return control 

def get_delta_x_and_c(x_min, x_max, c_min, c_max, c_prime_min, c_prime_max):
    """
    get flow and cost differentials
    
    Parameters
    ----------
    x_min : float
        total flow on minimum path
    x_max : float
        total flow on maximum path
    c_min : float 
        cost of minimum path
    c_max : float 
        cost of maximum path
    c_prime_min : float 
        derivative cost of minimum path
    c_prime_max : float 
        derivative cost of maximum path
    Returns
    -------
    del_x : float
        flow differential
    del_c : float 
        cost differential used for checking paths have equalised
    
    """
    # if the cost of c_min is less that c_max, potential shift is x_max
    # It is possible that during equalising of the costs, x_min and x_max will effectively swap signs.
    # i.e. we may overshoot the optimal flow
    if c_min < c_max:
        del_x = x_max
    else:
        del_x = x_min

    # in the case that all flow has been removed from the max path, then del_x is 0, 
    # i.e. we should remove no flow and there is no path cost differential
    if del_x <= 0:
        return (0,0)

    # in the case where the sum of the cost derivatives are zero, shift all flow from max to min
    if c_prime_min + c_prime_max <= 0:
        # check that paths haven't swapped signs
        if c_min < c_max:
            del_x = x_max
        else:
            del_x = -x_min
    else:
        # shift flow by the minimum available or the newton step
        del_x = min(del_x, (c_max-cmin)/(c_prime_max - c_prime_min))
    
    return del_x, np.abs(c_max-c_min)

def update_path_flow(del_x, k, j, alpha):

    """
    update path flows for paired alternative segments
    
    Parameters
    ----------
    del_x : float
        flow to shift
    k : int
        start node of path to j
    j : int
        end node of path
    alpha : dict    
        node predecessor-arc array
    Returns
    -------
    x_p : float
        path p's revised total flow. i.e. the minimum amount of flow for the path that could be moved
            to another path k->j to ensure flow conservation
    c_p : float
        path p's revised cost
    c_prime : float
        path p's revised cost derivative 
    """
    # note that this is implemented as it is written in Dial 1999
    global x
    x_p = np.inf
    c_p = 0
    c_prime = 0

    # set i = to the end node
    i = j
    while not i == k:
        # get arc indices
        u = i
        v = alpha[i]
        # update the flow of link u,v
        x[u,v] += del_x
        # compute the path revised total flow
        x_p = min(x_p, x[u,v])

        # compute the arc cost and derivative for use in computing the path cost and derivative
        c_uv = arc_cost(u,v,x[u,v])
        c_prime_uv = arc_derivative(u,v,x[u,v])

        # compute the path cost and derivative
        c_p += c_uv
        c_prime += c_prime_uv

    return x_p, c_p, c_prime

def arc_cost(i,j,x):
    return c_o[i,j]*(1+b[i,j]*(x/k[i,j])**n[i,j])

def arc_derivative(i,j,x):
    return n[i,j]*c_o[i,j]*b[i,j]/(k[i,j]**n[i,j])*x**(n[i,j]-1)