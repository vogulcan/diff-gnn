import utils

from math import sqrt

import torch
from torch.nn.parameter import Parameter

import networkx as nx
from matplotlib import pyplot as plt

class explainer:
    """
    GNNExplainer implementation from PyTorch Geometric - only for edge mask training
    
    
    For more info:
        https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/explain/algorithm/gnn_explainer.html#GNNExplainer
    
    and

    edge_mask in models.emb_model.forward()
    
    """
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'edge_ent': 1.0,
        'EPS': 1e-15,
    }
    
    def __init__(self, model, q_data):
        self.model = model
        self.query_data = q_data
        self.edge_mask = self._set_edge_mask(q_data)
        self.hard_edge_mask = None

    def _set_edge_mask(self, q_data):
        (N, F), E = q_data.x.size(), q_data.edge_index.size(1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = Parameter(torch.randn(E, device=utils.get_device()) * std)
        return edge_mask.to(utils.get_device())

    def _loss(self, pred, label):
        loss_f = torch.nn.NLLLoss()
        loss = loss_f(pred, label)
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()
        
        return loss

    def train_edge_masks(self, model, bs_, emb_as_, epoch=200):
        Parameters = [self.edge_mask]
        optimizer = torch.optim.Adam(Parameters, lr=.01)
        label = torch.tensor([0], device=utils.get_device())

        model.train()

        for i in range(epoch):
            optimizer.zero_grad()
            curr_b = model.emb_model(bs_.x, bs_.edge_index, bs_.edge_attr, bs_.batch, edge_mask = self.edge_mask.sigmoid())
            with torch.no_grad():
                pred = model.predict((emb_as_, curr_b))
            
            pred = model.clf_model(pred.unsqueeze(1))
            loss = self._loss(pred, label)
            loss.backward()

            if i == 0 and self.edge_mask is not None:                
                self.hard_edge_mask = self.edge_mask.grad != 0.0

        return self.edge_mask.sigmoid()
    


def visualize_edges(G, anchor=25):
    widths = nx.get_edge_attributes(G, 'edge_attr')
    for k, v in widths.items():
        widths[k] = v * 100

    nodelist = G.nodes
    node_color = ["red" if n == anchor else "blue" for n in nodelist]
    plt.figure(figsize=(12,8))

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G,pos,
                        nodelist=nodelist,
                        node_size=1500,
                        node_color=node_color,
                        alpha=0.7)
    nx.draw_networkx_edges(G,pos,
                        edgelist = widths.keys(),
                        width=list(widths.values()),
                        edge_color='lightblue',
                        alpha=0.6)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='white')
    plt.box(False)
    plt.show()