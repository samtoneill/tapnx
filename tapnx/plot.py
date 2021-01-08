import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def get_edge_colors_by_attr(
    G, attr, num_bins=None, cmap="plasma", start=0, stop=1, na_color="none", equal_size=False
):
    """
    Get colors based on edge attribute values.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    attr : string
        name of a numerical edge attribute
    num_bins : int
        if None, linearly map a color to each value. otherwise, assign values
        to this many bins then assign a color to each bin.
    cmap : string
        name of a matplotlib colormap
    start : float
        where to start in the colorspace
    stop : float
        where to end in the colorspace
    na_color : string
        what color to assign edges with missing attr values
    equal_size : bool
        ignored if num_bins is None. if True, bin into equal-sized quantiles
        (requires unique bin edges). if False, bin into equal-spaced bins.
    Returns
    -------
    edge_colors : pandas.Series
        series labels are edge IDs (u, v, key) and values are colors
    """
    vals = pd.Series(nx.get_edge_attributes(G, attr))
    return _get_colors_by_value(vals, num_bins, cmap, start, stop, na_color, equal_size)

def plot_graph(G, edge_color="#999999", edge_labels=False,  node_labels=True, node_size=2):
    """
    Plot a networkx graph
    Parameters
    ----------
    G : networkx.DiGraph
        input graph
    edge_color: (color or array of colors (default=’k’))
        Edge color. Can be a single color or a sequence of colors with the same length as edgelist. 
        Color can be string, or rgb (or rgba) tuple of floats from 0-1. If numeric values are specified 
        they will be mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    edge_labels: bool
        draw edge labels
    node_labels: bool
        draw node labels
    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.
    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis
    """

    fig, ax = plt.subplots()
    if 'pos' in G.graph:
        # draw the graph with positions
        nx.draw(G, pos=G.graph['pos'], node_size=node_size, edge_color=edge_color, ax=ax, with_labels=node_labels)
    else:
        nx.draw(G, node_size=node_size, edge_color=edge_color, ax=ax, with_labels=node_labels)

    if edge_labels:
        edge_labels = nx.get_edge_attributes(G,'id')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    return fig, ax

def _get_colors_by_value(vals, num_bins, cmap, start, stop, na_color, equal_size):
    """
    Map colors to the values in a series.
    Parameters
    ----------
    vals : pandas.Series
        series labels are node/edge IDs and values are attribute values
    num_bins : int
        if None, linearly map a color to each value. otherwise, assign values
        to this many bins then assign a color to each bin.
    cmap : string
        name of a matplotlib colormap
    start : float
        where to start in the colorspace
    stop : float
        where to end in the colorspace
    na_color : string
        what color to assign to missing values
    equal_size : bool
        ignored if num_bins is None. if True, bin into equal-sized quantiles
        (requires unique bin edges). if False, bin into equal-spaced bins.
    Returns
    -------
    color_series : pandas.Series
        series labels are node/edge IDs and values are colors
    """
    if len(vals) == 0:
        raise ValueError("There are no attribute values.")

    if num_bins is None:
        # calculate min/max values based on start/stop and data range
        vals_min = vals.dropna().min()
        vals_max = vals.dropna().max()
        full_range = (vals_max - vals_min) / (stop - start)
        full_min = vals_min - full_range * start
        full_max = full_min + full_range

        # linearly map a color to each attribute value
        normalizer = colors.Normalize(full_min, full_max)
        scalar_mapper = cm.ScalarMappable(normalizer, cm.get_cmap(cmap))
        color_series = vals.map(scalar_mapper.to_rgba)
        color_series.loc[pd.isnull(vals)] = na_color

    else:
        # otherwise, bin values then assign colors to bins
        cut_func = pd.qcut if equal_size else pd.cut
        bins = cut_func(vals, num_bins, labels=range(num_bins))
        bin_colors = get_colors(num_bins, cmap, start, stop)
        color_list = [bin_colors[b] if pd.notnull(b) else na_color for b in bins]
        color_series = pd.Series(color_list, index=bins.index)

    return color_series