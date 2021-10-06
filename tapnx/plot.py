"""Plot functions for TAPnx."""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

from . import utils_graph

def get_edge_colors_by_attr(
    G, attr, num_bins=None, cmap="plasma", start=0.1, stop=1, na_color="none", equal_size=False
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

def plot_graph(G, edge_color="#999999", edge_labels=False, edge_label_attr='id',
                node_labels=True, node_size=2,
                show=False, save=False, close=False, filepath=None, edge_list=None):
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
    edge_label_attr: string
        attribut to display on edge label
    node_labels: bool
        draw node labels
    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.
    show : bool
        if True, call pyplot.show() to show the figure
    close : bool
        if True, call pyplot.close() to close the figure
    save : bool
        if True, save the figure to disk at filepath
    filepath : string
        if save is True, the path to the file. file format determined from
        extension. if None, use settings.imgs_folder/image.png
    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis
    """

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos=G.graph['pos'], node_size=node_size)
    nx.draw_networkx_edges(G, pos=G.graph['pos'], edge_color=edge_color, edgelist=edge_list)
    # nx.draw(
    #         G, pos=G.graph['pos'], node_size=node_size, 
    #         edge_color=edge_color, 
    #         ax=ax, with_labels=node_labels
    # )
    
    if edge_labels:
        edge_labels = nx.get_edge_attributes(G,edge_label_attr)
        nx.draw_networkx_edge_labels(G, pos=G.graph['pos'], edge_labels=edge_labels)

    return fig, ax

def plot_nodes(G, pos, node_size= 2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size)
    return fig, ax

def plot_node_labels(G, pos, labels=None, font_size=12, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=font_size)
    return fig, ax

def plot_edge_labels(G, pos, edge_labels=None, ax=None, font_size=10):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=font_size)
    return fig, ax

def plot_edges(G, pos, edge_color="#999999", edge_labels=False, edge_label_attr='id', 
                edge_list=None, width=1, arrowsize=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    nx.draw_networkx_edges(G, pos=pos, edge_color=edge_color, edgelist=edge_list, width=width, arrowsize=arrowsize, ax=ax)

    if edge_labels:
        edge_labels = nx.get_edge_attributes(G,edge_label_attr)
        nx.draw_networkx_edge_labels(G, pos=G.graph['pos'], edge_labels=edge_labels)

    return fig, ax

def plot_graph_path(G, path, ax=None, **pg_kwargs):
    if ax is None:
        # plot the graph but not the route, and override any user show/close
        # args for now: we'll do that later
        override = {"show", "save", "close"}
        kwargs = {k: v for k, v in pg_kwargs.items() if k not in override}
        fig, ax = plot_graph(G, show=False, save=False, close=False, **kwargs)
    else:
        fig = ax.figure

    path_edges = utils_graph.edges_from_path(path)
    pos = G.graph['pos']

    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',ax=ax)
    
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

def get_colors(n, cmap="viridis", start=0.0, stop=1.0, alpha=1.0, return_hex=False):
    """
    Get `n` evenly-spaced colors from a matplotlib colormap.
    Parameters
    ----------
    n : int
        number of colors
    cmap : string
        name of a matplotlib colormap
    start : float
        where to start in the colorspace
    stop : float
        where to end in the colorspace
    alpha : float
        opacity, the alpha channel for the RGBa colors
    return_hex : bool
        if True, convert RGBa colors to HTML-like hexadecimal RGB strings. if
        False, return colors as (R, G, B, alpha) tuples.
    Returns
    -------
    color_list : list
    """
    color_list = [cm.get_cmap(cmap)(x) for x in np.linspace(start, stop, n)]
    if return_hex:
        color_list = [colors.to_hex(c) for c in color_list]
    else:
        color_list = [(r, g, b, alpha) for r, g, b, _ in color_list]
    return color_list

def draw_additional_labels(G, labels, pos, shift, ax, font_color='k'):
    
    pos_higher = {}
    for k, v in pos.items():
            # shift node to right and up or down depending on sign of shift
            pos_higher[k] = (v[0]+np.abs(shift), v[1]+shift)

    nx.draw_networkx_labels(G, pos_higher,labels, font_color=font_color, ax=ax)
    return ax