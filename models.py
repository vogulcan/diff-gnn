import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

# pl module
from torchmetrics import Accuracy
from utils import build_optimizer
import torch.optim as optim
import pytorch_lightning as pl

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
            hidden_input_dim = hidden_dim * (l + 1)
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
class model2explainer(nn.Module):
    def __init__(self, model, b, target_emb):
        super(model2explainer, self).__init__()
        self.model = model
        self.e = b.edge_attr
        self.batch = b.batch
        self.target_emb = target_emb

    def forward(self, x, edge_index):
        emb_b = self.model.emb_model(x, edge_index, self.e, self.batch)
        pred = self.model(self.target_emb, emb_b)
        with torch.no_grad():
            pred = self.model.predict(pred)
            val = pred.item()
        pred = self.model.clf_model(pred.unsqueeze(1))
        return pred, val
from torch_geometric.explain import Explainer, GNNExplainer, Explanation
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch import Tensor
from typing import Optional, Tuple, Union
class explainer(GNNExplainer):
    coeffs = {
        "edge_size": 0.005,
        "edge_reduction": "sum",
        "node_feat_size": 1.0,
        "node_feat_reduction": "mean",
        "edge_ent": 1.0,
        "node_feat_ent": 0.1,
        "EPS": 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.losses = []

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)
        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)

        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        optimizer.zero_grad(set_to_none=False)

        self.hard_node_mask = self.node_mask.grad != 0.0
        self.hard_edge_mask = self.edge_mask.grad != 0.0
        vals = []
        for i in range(self.epochs):
            optimizer.zero_grad()

            h = x if self.node_mask is None else x * self.node_mask.sigmoid()
            y_hat, val = model(h, edge_index, **kwargs)
            y = target
            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())
            print(val)
            vals.append(val)

            last = vals[-1]
            prev = vals[-2] if len(vals) > 1 else last
            if i > 0 and last > prev:
                if last - prev < 0.01:
                    print("early stopping")
                    break
            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not yet supported in "
                f"'{self.__class__.__name__}'"
            )

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        # edge_mask = self.edge_mask.sigmoid()
        # node_mask = self.node_mask.sigmoid()

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

class pl_model(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim

        self.margin = args.margin
        
        self.automatic_optimization=False

        self.emb_model = GNN_Pack(input_dim, hidden_dim, hidden_dim, args)
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

    def predict(self, pred):
        emb_as, emb_bs = pred

        e = torch.sum(
            torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_bs - emb_as)
            ** 2,
            dim=1,
        )
        return e

    def criterion(self, pred, labels):
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

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def training_step(self, data, batch_idx):
        emb_opt, clf_opt = self.optimizers()        
        emb_opt.zero_grad()
        scheduler = self.lr_schedulers()
        
        data = [i.squeeze(0) for i in data]
        emb_as = self.emb_model(data[0], data[1], data[2], data[3])
        emb_bs = self.emb_model(data[4], data[5], data[6], data[7])
        
        labels_per_group = self.args.batch_size // 2
        labels = torch.tensor([1] * labels_per_group + [0] * labels_per_group).to(utils.get_device())

        pred = self(emb_as, emb_bs)
        loss = self.criterion(pred, labels)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.emb_model.parameters(), 1.0)
        emb_opt.step()
        scheduler.step()
        with torch.no_grad():
            pred = self.predict(pred)
        self.clf_model.zero_grad()
        clf_opt.zero_grad()
        pred = self.clf_model(pred.unsqueeze(1))
        criterion = nn.NLLLoss()
        clf_loss = criterion(pred, labels)
        self.manual_backward(clf_loss)
        clf_opt.step()

        pred = pred.argmax(dim=1)
        self.train_acc(pred, labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True)

    def _shared_step(self, data):
        data = [i.squeeze(0) for i in data]
        emb_as = self.emb_model(data[0], data[1], data[2], data[3])
        emb_bs = self.emb_model(data[4], data[5], data[6], data[7])
        labels_per_group = self.args.batch_size // 2
        labels = torch.tensor([1] * labels_per_group + [0] * labels_per_group).to(utils.get_device())
        pred = self.predict(self(emb_as, emb_bs))
        pred = self.clf_model(pred.unsqueeze(1))
        pred = pred.argmax(dim=1)
        return pred, labels
        
    def validation_step(self, data, batch_idx):
        pred, labels = self._shared_step(data)
        self.val_acc(pred, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=True, on_epoch=True)
        return
    
    def test_step(self, data, batch_idx):    
        pred, labels = self._shared_step(data)
        self.test_acc(pred, labels)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=True, on_epoch=True)
        return

    def configure_optimizers(self):
        scheduler, emb_opt = build_optimizer(self.args, self.emb_model.parameters())
        clf_opt = optim.Adam(self.clf_model.parameters(), lr=self.args.lr)
        return [emb_opt, clf_opt], [scheduler]
