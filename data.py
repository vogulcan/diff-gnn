import utils
import torch
import torch_geometric.utils as pyg_utils
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset, Batch

import argparse

class _hic_dataset_torch(Dataset):
    def __init__(
        self,
        pyg_dataset: InMemoryDataset,
        args
    ):
        self.dataset = pyg_dataset
        self.batch_size = args.batch_size
        self.transform = None
        self.min_size = args.min_size
        self.max_size_Q = args.max_size_Q
        self.max_size_T = args.max_size_T
        self.edge_margin = args.edge_margin

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        a_batch, b_batch = self.gen_batch()
        a_x, a_edge_index, a_edge_attr, a_batch = (
            a_batch.x,
            a_batch.edge_index,
            a_batch.edge_attr,
            a_batch.batch,
        )
        b_x, b_edge_index, b_edge_attr, b_batch = (
            b_batch.x,
            b_batch.edge_index,
            b_batch.edge_attr,
            b_batch.batch,
        )
        return a_x, a_edge_index, a_edge_attr, a_batch, b_x, b_edge_index, b_edge_attr, b_batch

    def gen_batch(self):
        batch_size, min_size, max_size_q, max_size_t, edge_margin = (
            self.batch_size,
            self.min_size,
            self.max_size_Q,
            self.max_size_T,
            self.edge_margin,
        )

        categorical_div = [4, 4, 8, 8, 8, 8]
        a_graphs = []
        b_graphs = []

        for category_Id, div in enumerate(categorical_div):
            as_, bs_, _ = utils.synthesize_graphs(
                self.dataset,
                category_Id,
                batch_size // div,
                min_size,
                max_size_q,
                max_size_t,
                edge_margin,
            )
            a_graphs += as_
            b_graphs += bs_

        a_batch = Batch.from_data_list(
            [
                pyg_utils.from_networkx(
                    g, group_node_attrs=["x"], group_edge_attrs=["edge_attr"]
                )
                for g in a_graphs
            ]
        )
        a_batch.edge_attr = a_batch.edge_attr.type(torch.float32)

        b_batch = Batch.from_data_list(
            [
                pyg_utils.from_networkx(
                    g, group_node_attrs=["x"], group_edge_attrs=["edge_attr"]
                )
                for g in b_graphs
            ]
        )
        b_batch.edge_attr = b_batch.edge_attr.type(torch.float32)

        return a_batch, b_batch


class _hic_dataset_pyg(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


class _hic_datamodule_pl(LightningDataModule):
    def __init__(self, args: argparse.Namespace = None, steps_per_epoch = None):
        super().__init__()
        dataset = _hic_dataset_pyg(root=args.dataset).shuffle()
        self.test_dataset = _hic_dataset_torch(dataset[: len(dataset) // 10], args)
        self.val_dataset = _hic_dataset_torch(
            dataset[len(dataset) // 10 : 2 * len(dataset) // 10], args
        )
        self.train_dataset = _hic_dataset_torch(dataset[2 * len(dataset) // 10 :], args)
        
        self.batch_size = args.batch_size
        self.DataLoader_batch_size = 1
        self.num_workers = args.n_workers
        self.steps_per_epoch = steps_per_epoch

    def train_dataloader(self):
        return self._dl_wrapper(DataLoader(self.train_dataset, batch_size=self.DataLoader_batch_size, shuffle=False, num_workers=self.num_workers), n_steps=self.steps_per_epoch['train'])

    def val_dataloader(self):
        return self._dl_wrapper(DataLoader(self.val_dataset, batch_size=self.DataLoader_batch_size, shuffle=False, num_workers=self.num_workers), n_steps=self.steps_per_epoch['val'])

    def test_dataloader(self):
        return self._dl_wrapper(DataLoader(self.test_dataset, batch_size=self.DataLoader_batch_size, shuffle=False, num_workers=self.num_workers), n_steps=self.steps_per_epoch['test'])

    class _dl_wrapper():
        def __init__(self, data_loader, n_steps):
            self.step = n_steps
            self.idx = 0
            self.iter_loader = iter(data_loader)

        def __iter__(self):
            return self
        
        def __len__(self):
            return self.step
        
        def __next__(self):
            if self.idx == self.step:
                self.idx = 0
                raise StopIteration
            else:
                self.idx += 1
            try:
                return next(self.iter_loader)
            except StopIteration:
                self.iter_loader = iter(self.loader)
                return next(self.iter_loader)
        