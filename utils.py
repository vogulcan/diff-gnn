import torch
import torch.optim as optim

import networkx as nx
import numpy as np
import scipy.stats as stats
import math

from torch_geometric.utils import from_networkx, to_networkx

import random
import warnings
import sys


def monitor_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    return convert_size(f), convert_size(r), convert_size(a)


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay
        )
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate
        )
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart
        )
    return scheduler, optimizer


def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        anchor_node = [i[0] for i in graphs[0].nodes(data=True) if i[1]["x"][1] == 1][0]

        start_node = anchor_node
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


def find_closest(arr, val):
    return np.abs(arr - val).argmin()


def relabel_nodes(G, max_nodes):
    G = nx.convert_node_labels_to_integers(G)

    lin_centrality_attrs = np.linspace(
        -1, 1, max_nodes, endpoint=True
    )  # relabel nodes depending on centrality encoding
    mapping = {}
    for node_id, attr in dict(G.nodes(data=True)).items():
        mapping[node_id] = find_closest(lin_centrality_attrs, attr["x"][0])

    return nx.relabel_nodes(G, mapping, copy=True)


device_cache = None


def get_device():
    global device_cache
    if device_cache is None:
        device_cache = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # device_cache = torch.device("cpu")
    return device_cache


def constGraph(i, matrix, arm_range, attrs, minNodes):
    local_m = matrix[
        i - arm_range : i + arm_range + 1, i - arm_range : i + arm_range + 1
    ]
    G = nx.from_numpy_array(local_m)
    nx.set_node_attributes(G, attrs)

    if len(G.nodes()) == 0:
        return None
    else:
        largest_cc = max(nx.connected_components(G), key=len)

    if arm_range in largest_cc and len(largest_cc) > minNodes:
        S = G.subgraph(largest_cc).copy()
        S = nx.convert_node_labels_to_integers(S)
        return from_networkx(
            S,
            group_node_attrs=["centrality", "anchor"],
            group_edge_attrs=["weight"],
        )
    else:
        return None


def constGraphList(
    matrix, minNodes, maxNodesT, maxNodesQ=None, use_prcnt=10, idx_list=None
):
    graphs = []

    lin_centrality_attrs = np.linspace(-1, 1, maxNodesT, endpoint=True)
    anchor_attrs = [-1.0 for n in range(maxNodesT)]
    anchor_attrs[maxNodesT // 2] = 1.0

    if maxNodesQ is not None:
        arm_range = maxNodesQ // 2
        nDiff = (maxNodesT - maxNodesQ) // 2
        anchor_attrs = anchor_attrs[nDiff:-nDiff]
        lin_centrality_attrs = lin_centrality_attrs[nDiff:-nDiff]

        attrs = {
            n: {"centrality": lin_centrality_attrs[n], "anchor": anchor_attrs[n]}
            for n in range(maxNodesQ)
        }

    else:
        arm_range = maxNodesT // 2
        attrs = {
            n: {"centrality": lin_centrality_attrs[n], "anchor": anchor_attrs[n]}
            for n in range(maxNodesT)
        }

    if idx_list is not None:
        for i in idx_list:
            graphs.append(constGraph(i, matrix, arm_range, attrs, minNodes))
        return graphs

    else:
        idx_used = []
        idx_range = list(range(matrix.shape[0]))
        while len(graphs) < len(idx_range) // 100 * use_prcnt:
            if len(idx_range) == 0:
                break
            i = random.choice(idx_range)
            idx_range.remove(i)
            idx_used.append(i)

            G = constGraph(i, matrix, arm_range, attrs, minNodes)
            if G is not None:
                graphs.append(G)
            else:
                continue

        return graphs, idx_used


def add_and_remove_edges(G, p_new_connection, p_remove_connection):
    new_edges = []
    rem_edges = []
    for node in G.nodes():
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected]
        if len(unconnected):
            if random.random() < p_new_connection:
                new = random.choice(unconnected)
                G.add_edge(node, new)
                new_edges.append((node, new))
                unconnected.remove(new)
                connected.append(new)
        if len(connected):
            if random.random() < p_remove_connection:
                remove = random.choice(connected)
                G.remove_edge(node, remove)
                rem_edges.append((node, remove))
                connected.remove(remove)
                unconnected.append(remove)
    return rem_edges, new_edges


def check_graphs(graph, anchor):
    if anchor not in graph.nodes():
        warnings.warn(f"graph does not contain anchor")
    if not nx.is_connected(graph):
        warnings.warn(f"graph is not connected")


def choose_graph(graphs):
    """
    ps = np.array([g.x.shape[0] for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    idx = dist.rvs()
    graph = graphs[idx]
    graph = to_networkx(
            graph, node_attrs=["x"], edge_attrs=["edge_attr"]
        ).to_undirected().copy()
    """

    graph = to_networkx(
        random.choice(graphs),
        to_undirected=True,
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
    )
    return graph
    # return random.choice(graphs)


def sample_neigh(graph, size):
    anchor_node = [i[0] for i in graph.nodes(data=True) if i[1]["x"][1] == 1][0]
    start_node = anchor_node
    neigh = [start_node]
    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
    visited = set([start_node])
    while len(neigh) < size and frontier:
        new_node = random.choice(list(frontier))
        assert new_node not in neigh
        neigh.append(new_node)
        visited.add(new_node)
        frontier += list(graph.neighbors(new_node))
        frontier = [x for x in frontier if x not in visited]
    if len(neigh) == size:
        return graph, neigh


def uniq_nodes(max_size_Q, max_size_T):
    n_diff = (max_size_T - max_size_Q) // 2
    all_nodes = list(range(max_size_T))
    return all_nodes[:n_diff] + all_nodes[-n_diff:], all_nodes[n_diff:-n_diff]


def sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq):
    graph = choose_graph(graphs)
    graph = relabel_nodes(graph, max_size_T)

    b_uniq_graph = graph.copy()
    b_uniq_graph.remove_nodes_from(a_uniq)
    if b_uniq_graph.number_of_nodes() < min_size:
        return None
    b_size = random.randint(min_size, len(b_uniq_graph))
    b_uniq_graph, b = sample_neigh(b_uniq_graph, b_size)
    b_uniq_graph = b_uniq_graph.subgraph(b).copy()

    a_uniq_graph = graph.copy()
    a_uniq_graph.remove_nodes_from(b_uniq)
    if len(a_uniq_graph) <= 0:
        return None

    add_size = random.randint(1, len(a_uniq_graph))
    nodes_to_add = random.sample(sorted(a_uniq_graph.nodes), add_size)
    return graph.subgraph(b + nodes_to_add), b_uniq_graph, graph


def sample_different_region(graphs, min_size, max_size_T, a_uniq, b_uniq):
    a_graph = choose_graph(graphs)
    b_graph = choose_graph(graphs)
    a_graph = relabel_nodes(a_graph, max_size_T)
    b_graph = relabel_nodes(b_graph, max_size_T)
    b_graph.remove_nodes_from(a_uniq)

    if len(b_graph) > min_size:
        b_size = random.randint(min_size, len(b_graph))

    elif len(b_graph) < min_size:
        return None
    else:
        b_size = min_size

    b_graph, b = sample_neigh(b_graph, b_size)
    b_graph = b_graph.subgraph(b).copy()

    if len(a_graph) <= b_size:
        return None

    a_graph, a = sample_neigh(a_graph, random.randint(b_size + 1, len(a_graph)))
    a_graph = a_graph.subgraph(a).copy()
    return a_graph, b_graph


def synthesize_graphs(graphs, cat, n, min_size, max_size_Q, max_size_T, margin):
    a_, b_ = [], []

    if max_size_Q >= max_size_T:
        sys.exit("max_size_Q should be smaller than max_size_T")

    a_uniq, b_uniq = uniq_nodes(max_size_Q, max_size_T)

    trial = 0
    while len(a_) < n:
        trial += 1

        if cat == 0:  # graphs sampled from same region
            graph_tup = sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq)
            if graph_tup == None:
                continue
            else:
                a_graph, b_graph, _ = graph_tup
                a_.append(a_graph)
                b_.append(b_graph)

        elif (
            cat == 1
        ):  # graphs sampled from same region, edge feats changed within margin
            graph_tup = sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq)
            if graph_tup == None:
                continue
            else:
                a_graph, b_graph, _ = graph_tup
                edge_updates = np.random.uniform(-margin, margin, len(b_graph.edges))
                for edge, update in zip(list(b_graph.edges()), edge_updates):
                    val = b_graph[edge[0]][edge[1]]["edge_attr"]
                    b_graph[edge[0]][edge[1]]["edge_attr"] = np.clip(
                        val + update, 10e-6, 1
                    )

                a_.append(a_graph)
                b_.append(b_graph)

        elif cat == 2:  # graphs sampled from different region
            graph_pair = sample_different_region(
                graphs, min_size, max_size_T, a_uniq, b_uniq
            )
            if graph_pair == None:
                continue
            else:
                a_.append(graph_pair[0])
                b_.append(graph_pair[1])

        elif (
            cat == 3
        ):  # graphs sampled from same region, edge feats changed outside margin
            graph_tup = sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq)
            if graph_tup == None:
                continue
            else:
                a_graph, b_graph, _ = graph_tup

                n_edges = len(b_graph.edges)
                edge_updates = np.random.uniform(
                    -margin, margin, n_edges
                )  # first distort within margin
                edge_update_out_neg = np.random.uniform(-1, -margin, n_edges).tolist()
                edge_update_out_pos = np.random.uniform(margin, 1, n_edges).tolist()

                n_edges_to_update = random.randint(1, n_edges)
                edge_update_out = np.random.choice(
                    edge_update_out_neg + edge_update_out_pos, n_edges_to_update
                )
                edge_update_out_idx = np.random.choice(
                    list(range(0, n_edges)), n_edges_to_update, replace=False
                )
                edge_updates[edge_update_out_idx] = edge_update_out

                for edge, update in zip(b_graph.edges(), edge_updates):
                    val = b_graph[edge[0]][edge[1]]["edge_attr"]
                    b_graph[edge[0]][edge[1]]["edge_attr"] = np.clip(
                        val + update, 10e-6, 1
                    )

                a_.append(a_graph)
                b_.append(b_graph)

        elif cat == 4:  # edge perturbation (add edges) - same region
            graph_tup = sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq)
            if graph_tup == None:
                continue
            else:
                a_graph, b_graph, _ = graph_tup
                _, added_edges = add_and_remove_edges(b_graph, random.random(), 0)
                edge_updates = np.random.uniform(0, 1, len(added_edges))
                update_dict = {e: v for e, v in zip(added_edges, edge_updates)}
                nx.set_edge_attributes(b_graph, update_dict, "edge_attr")

                a_.append(a_graph)
                b_.append(b_graph)

        elif cat == 5:  # node perturbation (add nodes) - same region
            graph_tup = sample_same_region(graphs, min_size, max_size_T, a_uniq, b_uniq)
            if graph_tup == None:
                continue
            else:
                a_graph, b_graph, graph = graph_tup
                a = list(a_graph.nodes())
                b = list(b_graph.nodes())

                total_nodes = list(graph.nodes())
                poss_nodes = [n for n in total_nodes if (n not in a) & (n not in b)]
                nMax_toAdd = len(a) - len(b) - 1

                if nMax_toAdd > len(poss_nodes):
                    nMax_toAdd = len(poss_nodes)

                if nMax_toAdd <= 0 or len(poss_nodes) == 0:
                    continue

                nNodes_to_add = random.randint(1, nMax_toAdd)
                nodes_to_add = np.random.choice(
                    poss_nodes, nNodes_to_add, replace=False
                )

                a_.append(a_graph)
                b_.append(graph.subgraph(b + nodes_to_add.tolist()))

        else:
            sys.exit("Category value out of index !")

        if trial > 2 * n and trial % n == 0:
            print(
                f"Can not build subgraphs effectively\n Cat ID:{cat}\n Trial:{trial} for nGraphs:{n}",
                flush=True,
            )

    return a_, b_, cat
