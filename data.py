import utils

import torch
from torch_geometric.data import InMemoryDataset
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Batch

import random
import tqdm

import sys

class _hic(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def load_dataset(dataset_path, split = 0.8):
    dataset = _hic(root=dataset_path)

    train, test = [], []
    dataset = list(dataset)
    random.shuffle(dataset)

    train_len = int(split * len(dataset)) # 80% train, 20% test
    return dataset[:train_len], dataset[train_len:]


class DataSource:
    def __init__(self, args, test_points=False):
        self.dataset = load_dataset(args.dataset)
        self.min_size = args.min_size
        self.max_size_Q = args.max_size_Q
        self.max_size_T = args.max_size_T

        self.edge_margin = args.edge_margin

    def gen_data_loaders(self, size, batch_size):
        loaders = [[batch_size] * (size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, train=True):
        batch_size = a
        min_size, max_size_q, max_size_t = self.min_size, self.max_size_Q, self.max_size_T 
        edge_margin = self.edge_margin

        train_set, test_set = self.dataset
        graphs = train_set if train else test_set

        categorical_div = [4, 4, 8, 8, 8, 8]
        a_graphs = []
        b_graphs = []

        for category_Id, div in enumerate(categorical_div):
            as_, bs_ = utils.synthesize_graphs(
                graphs,
                category_Id,
                batch_size // div,
                min_size,
                max_size_q,
                max_size_t,
                edge_margin,
            )
            a_graphs += as_
            b_graphs += bs_

        labels_per_group = batch_size // 2

        labels = torch.tensor([1] * labels_per_group + [0] * labels_per_group).to(
            utils.get_device()
        )

        a_batch = Batch.from_data_list(
            [
                pyg_utils.from_networkx(
                    g, group_node_attrs=["x"], group_edge_attrs=["edge_attr"]
                )
                for g in a_graphs
            ]
        )
        a_batch.edge_attr = a_batch.edge_attr.type(torch.float32)
        a_batch = a_batch.to(utils.get_device())

        b_batch = Batch.from_data_list(
            [
                pyg_utils.from_networkx(
                    g, group_node_attrs=["x"], group_edge_attrs=["edge_attr"]
                )
                for g in b_graphs
            ]
        )
        b_batch.edge_attr = b_batch.edge_attr.type(torch.float32)
        b_batch = b_batch.to(utils.get_device())

        return a_batch, b_batch, labels
