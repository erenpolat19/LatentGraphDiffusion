import torch
import torch_geometric
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
import random
import logging

import imp
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.config import cfg


class DataListSet(InMemoryDataset):
    def __init__(self, datalist):
        super().__init__()
        self.data, self.slices = self.collate(datalist)


# TODO: be careful about the shape; perhaps correct for zinc, but not for all
def add_virtual_node_edge(data, format):
    N = data.num_nodes
    data.num_node_per_graph = torch.tensor((N+1), dtype=torch.long)
    if format == 'PyG-ZINC':
        data.x_original = data.x + 1  # TODO: note that this is different across different datasets, whether dimension or +1
        data.edge_index_original = data.edge_index
        data.edge_attr_original = data.edge_attr
        # print(data.x, data.edge_attr)
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr,  # TODO: note that this is different across different datasets, whether dimension or +1
                         sparse_sizes=(N+1, N+1)).coalesce().to_dense()
        A[:, -1] = cfg.dataset.edge_encoder_num_types + 1
        A[-1, :] = cfg.dataset.edge_encoder_num_types + 1
        A[-1, -1] = 0
        A += torch.diag_embed(torch.ones([N+1], dtype=torch.long) * (cfg.dataset.edge_encoder_num_types + 4))
        edge_attr = A.reshape(-1, 1).long()
        data.edge_attr = edge_attr
        adj = torch.ones([N+1, N+1], dtype=torch.long)
        edge_index = dense_to_sparse(adj)[0]
        data.edge_index = edge_index
        data.x = torch.cat([data.x + 1, torch.ones([1, 1], dtype=torch.long) * (cfg.dataset.node_encoder_num_types + 1)], dim=0)
    elif format == 'OGB' and cfg.train.pretrain.atom_bond_only:  # TODO: for pretrain, only use the atom and bond type
        data.x_original = data.x[:, 0].unsqueeze(1)  # TODO: note that this is different across different datasets, whether dimension or +1
        data.edge_index_original = data.edge_index
        data.edge_attr_original = data.edge_attr
        # print(data.x, data.edge_attr)
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr[:, 0] + 1,
                         # TODO: note that this is different across different datasets, whether dimension or +1
                         sparse_sizes=(N + 1, N + 1)).coalesce().to_dense()
        A[:, -1] = cfg.dataset.edge_encoder_num_types + 1
        A[-1, :] = cfg.dataset.edge_encoder_num_types + 1
        A[-1, -1] = 0
        A += torch.diag_embed(torch.ones([N + 1], dtype=torch.long) * (cfg.dataset.edge_encoder_num_types + 4))
        edge_attr = A.reshape(-1, 1).long()
        data.edge_attr = edge_attr
        adj = torch.ones([N + 1, N + 1], dtype=torch.long)
        edge_index = dense_to_sparse(adj)[0]
        data.edge_index = edge_index
        data.x = torch.cat(
            [data.x[:, 0].unsqueeze(1), torch.ones([1, 1], dtype=torch.long) * (cfg.dataset.node_encoder_num_types + 1)], dim=0)
    elif format == 'OGB' and not cfg.train.pretrain.atom_bond_only:  # TODO: for pretrain, only use the atom and bond type
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        data.x_original = data.x  # TODO: note that this is different across different datasets, whether dimension or +1
        data.edge_index_original = data.edge_index
        data.edge_attr_original = data.edge_attr
        # print(data.x, data.edge_attr)
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr + 1,
                         # TODO: note that this is different across different datasets, whether dimension or +1
                         sparse_sizes=(N + 1, N + 1)).coalesce().to_dense()
        edge_virtual = torch.zeros([1, data.edge_attr.shape[1]], dtype=torch.long)
        for i, dim in enumerate(get_atom_feature_dims()):
            edge_virtual[0, i] = dim + 1
        A[:, -1] = edge_virtual.repeat(N+1, 1)
        A[-1, :] = edge_virtual.repeat(N+1, 1)
        for j in range(N + 1):
            A[j, j] = edge_virtual + 4
        edge_attr = A.reshape(N+1, N+1, -1).long()
        data.edge_attr = edge_attr
        adj = torch.ones([N + 1, N + 1], dtype=torch.long)
        edge_index = dense_to_sparse(adj)[0]
        data.edge_index = edge_index

        x_virtual = torch.zeros([1, data.x.shape[1]], dtype=torch.long)
        for i, dim in enumerate(get_atom_feature_dims()):
            x_virtual[0, i] = dim
        data.x = torch.cat([data.x, x_virtual], dim=0)

    elif format == 'PyG-QM9':  # TODO: finish QM9
        print('Already done in pretransform of QM9')
        pass
    else:
        raise NotImplementedError
    return data


# TODO: batchify large graphs, the code is as follows
# YourDataset(pre_transform=T.RootedEgoNets(hop=3))


# def random_mask(data, proportion_node=0.1, proportion_edge=0.1):
#     N = data.num_nodes
#     m = data.edge_index.shape[1]
#     masked_node_idx = random.sample(range(N), int(N * proportion_node))
#     masked_edge_idx = random.sample(range(m), int(m * proportion_edge))
#     data.x_unmasked = data.x
#     data.edge_attr_unmasked = data.edge_attr
#     data.x[masked_node_idx] = torch.ones([int(N * proportion_node), 1], dtype=torch.long, device=data.x.device) * (cfg.dataset.node_encoder_num_types + 2)
#     data.edge_attr[masked_edge_idx] = torch.ones([int(m * proportion_edge), 1], dtype=torch.long, device=data.x.device) * (cfg.dataset.edge_encoder_num_types + 1)
#     return data, masked_node_idx, masked_edge_idx

def random_mask(data, proportion_node=0.1, proportion_edge=0.1):
    N = data.num_nodes
    m = data.edge_index.shape[1]
    masked_node_idx = random.sample(range(N), int(N * proportion_node))
    masked_edge_idx = random.sample(range(m), int(m * proportion_edge))
    data.x_unmasked = data.x
    data.edge_attr_unmasked = data.edge_attr
    data.x[masked_node_idx] = torch.ones([int(N * proportion_node), 1], dtype=data.x.dtype, device=data.x.device) * (cfg.dataset.node_encoder_num_types + 2)
    data.edge_attr[masked_edge_idx] = torch.ones([int(m * proportion_edge), 1], dtype=data.edge_attr.dtype, device=data.x.device) * (cfg.dataset.edge_encoder_num_types + 1)
    return data, masked_node_idx, masked_edge_idx
