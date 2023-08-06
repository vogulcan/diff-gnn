import utils
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(Embedder, self).__init__()
        self.emb_model = GNN_Pack(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if a is a subgraph of b (batched)

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(
            torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_bs - emb_as)
            ** 2,
            dim=1,
        )
        return e

    def criterion(self, pred, labels):
        """Loss function for order emb.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(
            torch.max(
                torch.zeros_like(emb_as, device=utils.get_device()), emb_bs - emb_as
            )
            ** 2,
            dim=1,
        )

        margin = self.margin
        e[labels == 0] = torch.max(
            torch.tensor(0.0, device=utils.get_device()), margin - e
        )[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss


class GNN_Pack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN_Pack, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        conv_model = self.build_conv_model(model_type=args.conv_type, edge_dim=1)
        self.convs = nn.ModuleList()

        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers, self.n_layers))

        for l in range(args.n_layers):
            if args.skip == "all" or args.skip == "learnable":
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim

            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        # self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, edge_dim):
        if model_type == "GINE":
            return lambda i, h: pyg_nn.GINEConv(
                nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)),
                edge_dim=edge_dim,
            )
        else:
            print("unrecognized model type")

    def forward(self, x, edge_index, e, batch, edge_mask=None):
        # x, e, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
        x = self.pre_mp(x)
        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs)):
            skip_vals = self.learnable_skip[i, : i + 1].unsqueeze(0).unsqueeze(-1)
            curr_emb = all_emb * torch.sigmoid(skip_vals)
            curr_emb = curr_emb.view(x.size(0), -1)
            if edge_mask is not None:
                x = self.convs[i](
                    curr_emb, edge_index, edge_attr=e * edge_mask.view(-1, 1)
                )
            else:
                x = self.convs[i](curr_emb, edge_index, edge_attr=e)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)

        # emb = self.batch_norm(emb)   # TODO: test
        # out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
