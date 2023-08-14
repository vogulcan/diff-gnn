import utils
import torch
import torch_geometric.utils as pyg_utils
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch_geometric.data import InMemoryDataset, Batch

import argparse
import tqdm
import sys

import numpy as np
from itertools import chain


class _hic_dataset_torch(Dataset):
    def __init__(self, pyg_dataset, args, state="train"):
        self.dataset = pyg_dataset
        self.batch_size = args.batch_size
        self.transform = None
        self.min_size = args.min_size
        self.max_size_Q = args.max_size_Q
        self.max_size_T = args.max_size_T
        self.edge_margin = args.edge_margin

        self.args = args

        self.state = state

        if self.state in ["test", "val"]:
            n_step = vars(args)[f"steps_per_{state}"]
            self.dataset = [
                self.gen_batch()
                for i in tqdm.tqdm(
                    range(n_step),
                    desc=f"Generating {state} data",
                    total=n_step,
                    file=sys.stdout,
                )
            ]

    def __len__(self):
        if self.state in ["test", "val"]:
            return len(self.dataset)
        else:
            return self.args.steps_per_train

    def __getitem__(self, idx):
        if self.state in ["test", "val"]:
            a_batch, b_batch = self.dataset[idx]
        elif self.state == "train":
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
        return (
            a_x,
            a_edge_index,
            a_edge_attr,
            a_batch,
            b_x,
            b_edge_index,
            b_edge_attr,
            b_batch,
        )

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
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        dataset = _hic_dataset_pyg(root=args.dataset).shuffle()
        self.test_dataset = _hic_dataset_torch(
            dataset[: len(dataset) // 10], args, state="test"
        )
        self.val_dataset = _hic_dataset_torch(
            dataset[len(dataset) // 10 : 2 * len(dataset) // 10], args, state="val"
        )
        self.train_dataset = _hic_dataset_torch(
            dataset[2 * len(dataset) // 10 :], args, state="train"
        )

        self.batch_size = args.batch_size
        self.num_workers = args.n_workers
        self.args = args

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
        )


class _cool_datamodule_pl(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.predict_dataset = _hic_dataset_cool(args)
        self.num_workers = args.n_workers

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            num_workers=self.num_workers,
        )


class _hic_dataset_cool(Dataset):
    def __init__(self, args):
        self.batch_size, self.min_size, self.max_size_q, self.max_size_t = (
            args.batch_size,
            args.min_size,
            args.max_size_Q,
            args.max_size_T,
        )
        self.comp_names = [args.sample1, args.sample2]
        self.comp = f"{self.comp_names[0]}_vs_{self.comp_names[1]}"

        self.npz = np.load(args.npz_path)
        self.chr_names = [file.replace(f'{args.sample1}-', '') for file in self.npz.files if args.sample1 in file]

        self.files = [
            [self.npz[file] for file in self.npz.files if args.sample1 in file],
            [self.npz[file] for file in self.npz.files if args.sample2 in file],
        ]
        
        assert len(self.files[0]) == len(self.files[1])
        assert [file.shape for file in self.files[0]] == [
            file.shape for file in self.files[1]
        ]

        self.loaders = self.prepare_loaders(self.files[0])
        self.chunk_indexer = [
            (chr_idx, chunk_idx)
            for chr_idx, loader in enumerate(self.loaders)
            for chunk_idx, _ in enumerate(loader)
        ]

        self.args = args

    def __len__(self):
        return len(self.chunk_indexer)

    def __getitem__(self, idx):
        chr_idx, chunk_idx = self.chunk_indexer[idx]
        idx_list = self.loaders[chr_idx][chunk_idx]

        target_matrix = self.files[0][chr_idx]
        query_matrix = self.files[1][chr_idx]

        target_graphs = utils.constGraphList(
            matrix=target_matrix,
            minNodes=self.min_size,
            maxNodesT=self.max_size_t,
            idx_list=idx_list,
        )

        query_graphs = utils.constGraphList(
            matrix=query_matrix,
            minNodes=self.min_size,
            maxNodesT=self.max_size_t,
            maxNodesQ=self.max_size_q,
            idx_list=idx_list,
        )

        target_graphs, query_graphs, present_idx = self.check_nones(
            target_graphs, query_graphs
        )

        chr_idx, chunk_idx = torch.tensor([chr_idx]), torch.tensor([chunk_idx])
        if target_graphs == [] or query_graphs == []:
            return self.dummy((chr_idx, chunk_idx, present_idx))

        target_batch = self.batch_graphs(target_graphs)
        query_batch = self.batch_graphs(query_graphs)

        return (
            target_batch.x,
            target_batch.edge_index,
            target_batch.edge_attr,
            target_batch.batch,
            query_batch.x,
            query_batch.edge_index,
            query_batch.edge_attr,
            query_batch.batch,
            chr_idx,
            chunk_idx,
            present_idx,
            torch.tensor([0]),
        )

    def prepare_loaders(self, files):
        arm_size = self.min_size // 2
        shapes = [file.shape[0] for file in files]
        ranges = [np.arange(arm_size, shape - arm_size) for shape in shapes]
        idx_ = [np.arange(0, _range.shape[0], self.batch_size)[1:] for _range in ranges]
        loaders = [np.array_split(loader, idx) for loader, idx in zip(ranges, idx_)]
        return loaders

    def flip(self):
        self.files.reverse()
        self.comp_names.reverse()
        print(f"Flipped to {self.comp_names[0]}_vs_{self.comp_names[1]}")
        self.comp = f"{self.comp_names[0]}_vs_{self.comp_names[1]}"

    def dummy(self, idx_tracker):
        x = torch.zeros(1, self.args.input_dim, dtype=torch.float32)
        edge_index = torch.zeros(2, 1, dtype=torch.int64)
        edge_attr = torch.zeros(1, 1, dtype=torch.float32)
        batch = torch.zeros(1, dtype=torch.int64)
        return (
            x,
            edge_index,
            edge_attr,
            batch,
            x,
            edge_index,
            edge_attr,
            batch,
            idx_tracker[0],
            idx_tracker[1],
            idx_tracker[2],
            torch.tensor([1]),
        )

    def batch_graphs(self, graphs):
        batch = Batch.from_data_list(graphs)
        batch.edge_attr = batch.edge_attr.type(torch.float32)
        batch.x = batch.x.type(torch.float32)
        return batch

    def check_nones(self, target_graphs, query_graphs):
        target_none_idx = [i for i, graph in enumerate(target_graphs) if graph is None]
        query_none_idx = [i for i, graph in enumerate(query_graphs) if graph is None]
        none_idx = list(set(target_none_idx + query_none_idx))
        all_idx = list(range(self.batch_size))
        present_idx = list(set(all_idx) - set(none_idx))

        target_graphs = [
            graph for i, graph in enumerate(target_graphs) if i in present_idx
        ]
        query_graphs = [
            graph for i, graph in enumerate(query_graphs) if i in present_idx
        ]

        return target_graphs, query_graphs, torch.tensor(present_idx)
