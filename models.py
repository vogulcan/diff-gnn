import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

# pl module
from torchmetrics import Accuracy
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
from utils import build_optimizer
import torch.optim as optim
import pytorch_lightning as pl


class pl_model_predict(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        input_dim = args.input_dim
        hidden_dim = args.hidden_dim
        self.emb_model = GNN_Pack(input_dim, hidden_dim, hidden_dim, args)
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))
        self.method = args.method

    def forward(self, pred):
        return (
            self.clf_model(self.score(pred)).argmax(dim=1)
            if self.method == "clf"
            else self.score(pred)
        )

    def score(self, pred):
        emb_as, emb_bs = pred
        e = torch.sum(
            torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_bs - emb_as)
            ** 2,
            dim=1,
        )
        return e

    def predict_step(self, data, batch_idx):
        data = [i.squeeze(0) for i in data]
        emb_as = self.emb_model(data[0], data[1], data[2], data[3])
        emb_bs = self.emb_model(data[4], data[5], data[6], data[7])
        pred = (emb_as, emb_bs)
        pred = self(pred)

        chr_idx = data[8]
        chunk_idx = data[9]
        present_idx = data[10]
        dummy = data[11]

        return {
            "pred": pred,
            "metadata": [chr_idx, chunk_idx, present_idx],
            "dummy": dummy,
        }


class pl_model(pl.LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.args = args
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim

        self.margin = args.margin

        self.automatic_optimization = False

        self.emb_model = GNN_Pack(input_dim, hidden_dim, hidden_dim, args)
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

        self.validation_step_outputs = {"pred": [], "raw_pred": [], "labels": []}
        self.test_step_outputs = {"pred": [], "raw_pred": [], "labels": []}

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
        labels = torch.tensor([1] * labels_per_group + [0] * labels_per_group).to(
            utils.get_device()
        )

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

        self.train_acc(pred.argmax(dim=1), labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True
        )
        self.logger.experiment.add_scalar(f"train_loss", loss.item(), self.global_step)

    def _shared_step(self, data):
        data = [i.squeeze(0) for i in data]

        emb_as = self.emb_model(data[0], data[1], data[2], data[3])
        emb_bs = self.emb_model(data[4], data[5], data[6], data[7])
        labels_per_group = self.args.batch_size // 2
        labels = torch.tensor([1] * labels_per_group + [0] * labels_per_group).to(
            utils.get_device()
        )
        raw_pred = self.predict(self(emb_as, emb_bs))
        pred = self.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
        return pred, raw_pred * -1, labels

    def validation_step(self, data, batch_idx):
        pred, raw_pred, labels = self._shared_step(data)
        self.validation_step_outputs["pred"].append(pred.cpu())
        self.validation_step_outputs["raw_pred"].append(raw_pred.cpu())
        self.validation_step_outputs["labels"].append(labels.cpu())

    def test_step(self, data, batch_idx):
        pred, raw_pred, labels = self._shared_step(data)
        self.test_step_outputs["pred"].append(pred.cpu())
        self.test_step_outputs["raw_pred"].append(raw_pred.cpu())
        self.test_step_outputs["labels"].append(labels.cpu())

    def _shared_compute_metrics(self, outputs, state):
        pred = torch.cat(outputs["pred"], dim=-1)
        labels = torch.cat(outputs["labels"], dim=-1)
        raw_pred = torch.cat(outputs["raw_pred"], dim=-1)

        if state == "val":
            self.val_acc(pred, labels)
            self.log(
                "val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True
            )
            acc = self.val_acc.compute()
        else:
            acc = torch.mean((pred == labels).type(torch.float))

        prec = (
            torch.sum(pred * labels).item() / torch.sum(pred).item()
            if torch.sum(pred) > 0
            else float("NaN")
        )
        recall = (
            torch.sum(pred * labels).item() / torch.sum(labels).item()
            if torch.sum(labels) > 0
            else float("NaN")
        )
        auroc = roc_auc_score(labels, raw_pred)
        avg_prec = average_precision_score(labels, raw_pred)
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

        metrics = {
            "acc": acc,
            "prec": prec,
            "recall": recall,
            "auroc": auroc,
            "avg_prec": avg_prec,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

        for k, v in metrics.items():
            if k == "val_acc":
                continue
            else:
                self.logger.experiment.add_scalar(f"{state}_{k}", v, self.current_epoch)

    def on_validation_epoch_end(self):
        self._shared_compute_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs = {"pred": [], "raw_pred": [], "labels": []}

    def on_test_epoch_end(self):
        self._shared_compute_metrics(self.test_step_outputs, "test")
        self.test_step_outputs = {"pred": [], "raw_pred": [], "labels": []}

    def configure_optimizers(self):
        scheduler, emb_opt = build_optimizer(self.args, self.emb_model.parameters())
        clf_opt = optim.Adam(self.clf_model.parameters(), lr=self.args.lr)
        return [emb_opt, clf_opt], [scheduler]


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
