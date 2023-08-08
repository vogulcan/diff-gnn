import utils

import networkx as nx
from matplotlib import pyplot as plt

from torch_geometric.utils import to_networkx
import numpy as np


def visualize_edges(G, ax=None, anchor=25):
    widths = nx.get_edge_attributes(G, "edge_attr")
    for k, v in widths.items():
        widths[k] = v * 100

    nodelist = G.nodes
    node_color = ["red" if n == anchor else "blue" for n in nodelist]
    print(nodelist)
    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist,
        node_size=1500,
        node_color=node_color,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=widths.keys(),
        width=list(widths.values()),
        edge_color="green",
        alpha=0.6,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos=pos, labels=dict(zip(nodelist, nodelist)), font_color="white", ax=ax
    )

    # plt.show()


def node_diff(t, q):
    t = t.to_directed() if not nx.is_directed(t) else t
    q = q.to_directed() if not nx.is_directed(q) else q
    return set(t.nodes()) - set(q.nodes()), set(q.nodes()) - set(t.nodes())


def edge_comp(t, q, q_max=41, t_max=51):
    t = t.to_directed() if not nx.is_directed(t) else t
    q = q.to_directed() if not nx.is_directed(q) else q
    t_copy = t.copy()

    n, _ = utils.uniq_nodes(q_max, t_max)
    t_copy.remove_nodes_from(n)

    t_spec = set(t_copy.edges()) - set(q.edges())
    t_spec_dict = dict()
    for edge in t_spec:
        t_spec_dict.update({edge: t_copy.edges[edge]["edge_attr"]})

    q_spec_dict = dict()
    q_spec = set(q.edges()) - set(t_copy.edges())
    for edge in q_spec:
        q_spec_dict.update({edge: q.edges[edge]["edge_attr"]})

    intersection = list(set(t_copy.edges).intersection(q.edges()))
    intersection_dict = dict()
    for edge in intersection:
        intersection_dict.update(
            {edge: t_copy.edges[edge]["edge_attr"] - q.edges[edge]["edge_attr"]}
        )

    return t_spec_dict, q_spec_dict, intersection_dict


def viz_attrs(edge_diff_dict, margin=0.15, anchor=None):
    checked = []
    vals = []
    return_dict = dict()
    for edge in edge_diff_dict:
        val = edge_diff_dict[edge]
        if np.abs(val) >= margin and set(edge) not in checked:
            if anchor is not None:
                if anchor in edge:
                    vals.append(val)
                    return_dict.update({edge: val})
            else:
                vals.append(val)
                return_dict.update({edge: val})
            checked.append(set(edge))

    return vals, return_dict


def apply_threshold(b, edge_mask, perc):
    threshold = np.percentile(edge_mask, perc)

    preserved_edges = b.edge_index.numpy()[:, np.where(edge_mask >= threshold)[0]]
    preserved_edges = [(int(edge[0]), int(edge[1])) for edge in preserved_edges.T]

    G_b = to_networkx(b, to_undirected=True, node_attrs=["x"], edge_attrs=["edge_attr"])
    drop_edges = [edge for edge in G_b.edges() if edge not in preserved_edges]
    G_b.remove_edges_from(drop_edges)

    return G_b
