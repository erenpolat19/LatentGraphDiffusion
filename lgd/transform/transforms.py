import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from torch_geometric.graphgym.config import cfg


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def generate_splits(data, g_split):
    n_nodes = len(data.x)
    train_mask = torch.zeros(n_nodes, dtype=bool)
    valid_mask = torch.zeros(n_nodes, dtype=bool)
    test_mask = torch.zeros(n_nodes, dtype=bool)
    idx = torch.randperm(n_nodes)
    val_num = test_num = int(n_nodes * (1 - g_split) / 2)
    train_mask[idx[val_num + test_num:]] = True
    valid_mask[idx[:val_num]] = True
    test_mask[idx[val_num:val_num + test_num]] = True
    data.train_mask = train_mask
    data.val_mask = valid_mask
    data.test_mask = test_mask
    return data


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data

def move_node_feat_to_x(data):
    """For ogbn-proteins, move the attribute node_species to attribute x."""
    data.x = data.node_species
    return data

def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data


def preprocess_edge_inductive(data):
    N = data.num_nodes
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)
    # data.edge_index_original = data.edge_index
    # data.edge_attr_original = data.edge_attr
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.ones(data.edge_index.shape[1]),
                       sparse_sizes=(N, N)).coalesce().to_dense()
    data.edge_label = torch.tensor(adj.reshape(-1).bool(), dtype=torch.long).reshape(-1)
    data.pos_edge_idx = torch.nonzero(data.edge_label, as_tuple=True)[0]
    data.neg_edge_idx = torch.nonzero(1 - data.edge_label, as_tuple=True)[0]
    assert data.pos_edge_idx.shape[0] + data.neg_edge_idx.shape[0] == N * N
    return data


def add_virtual_node_edge(data, format):
    N = data.num_nodes
    data.num_node_per_graph = torch.tensor((N+1), dtype=torch.long)
    data.edge_index_original = data.edge_index
    data.edge_attr_original = data.edge_attr
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.ones(data.edge_index.shape[1]),
                       sparse_sizes=(N + 1, N + 1)).coalesce().to_dense()
    data.original_edge = adj.reshape(-1).bool()

    # pestat_node = []
    # pestat_edge = []
    # for pename in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'rrwp']:
    #     pestat_var = f"pestat_{pename}"
    #     if data.get(pestat_var, None) is not None:
    #         pestat_node.append(getattr(data, pestat_var))
    # if data.get('rrwp', None) is not None:
    #     pestat_node.append(data.rrwp)
    # if pestat_node:
    #     pestat_node = torch.cat(pestat_node, dim=-1)
    #     pestat_node = torch.cat([pestat_node, torch.zeros([1, pestat_node.shape[1]], dtype=torch.float)], dim=0)
    #     data.pestat_node = pestat_node
    #
    # for pename in ['rrwp_edge', 'RD_val']:
    #     if data.get(pename, None) is not None:
    #         pestat_edge.append(getattr(data, pename))
    # if pestat_edge:
    #     pestat_edge = torch.cat(pestat_edge, dim=-1)
    #     pestat_edge_padded = torch.zeros([(N+1), (N+1), pestat_edge.shape[-1]], dtype=torch.float)
    #     pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
    #     data.pestat_edge = pestat_edge_padded.reshape((N+1)*(N+1), -1)

    if data.get('log_deg', None) is not None:
        data.log_deg = torch.cat([data.log_deg, torch.log(torch.tensor([N], dtype=torch.float))], dim=0)
    if data.get('deg', None) is not None:
        data.log_deg = torch.cat([data.deg, torch.tensor([N], dtype=torch.long)], dim=0)

    if format == 'PyG-ZINC':
        data.x_original = data.x + 1  # TODO: note that this is different across different datasets, whether dimension or +1
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
        # print(data.x, data.edge_attr)
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr + 1,
                         # TODO: note that this is different across different datasets, whether dimension or +1
                         sparse_sizes=(N + 1, N + 1, data.edge_attr.shape[1])).coalesce().to_dense()
        edge_virtual = torch.zeros([1, data.edge_attr.shape[1]], dtype=torch.long)
        for i, dim in enumerate(get_bond_feature_dims()):
            edge_virtual[0, i] = dim + 1
        A[:, -1] = edge_virtual.repeat(N+1, 1)
        A[-1, :] = edge_virtual.repeat(N+1, 1)
        for j in range(N + 1):
            A[j, j] = edge_virtual + 4
        edge_attr = A.reshape((N+1) * (N+1), data.edge_attr.shape[1]).long()
        data.edge_attr = edge_attr
        adj = torch.ones([N + 1, N + 1], dtype=torch.long)
        edge_index = dense_to_sparse(adj)[0]
        data.edge_index = edge_index

        x_virtual = torch.zeros([1, data.x.shape[1]], dtype=torch.long)
        for i, dim in enumerate(get_atom_feature_dims()):
            x_virtual[0, i] = dim
        data.x = torch.cat([data.x, x_virtual], dim=0)

    elif format == 'PyG-QM9':  # have finished in pretransform for QM9 (generation and regression)
        pass
    else:
        raise NotImplementedError
    data.num_nodes = N + 1
    return data


def no_virtual_node_edge(data, format):
    N = data.num_nodes
    data.num_node_per_graph = torch.tensor((N), dtype=torch.long)
    data.edge_index_original = data.edge_index.clone()
    if data.get('edge_attr', None) is None:
        data.edge_attr = torch.ones([data.edge_index.shape[1]], dtype=torch.long)
    data.edge_attr_original = data.edge_attr.clone()
    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=torch.ones(data.edge_index.shape[1]),
                       sparse_sizes=(N, N)).coalesce().to_dense()
    data.original_edge = adj.reshape(-1).bool()

    if len(data.edge_attr.shape) == 2:
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr,  # start from 1 to 4
                         sparse_sizes=(N, N, data.edge_attr.shape[1])).coalesce().to_dense()
        edge_attr = A.reshape(-1, data.edge_attr.shape[1])
    else:
        A = SparseTensor(row=data.edge_index[0],
                         col=data.edge_index[1],
                         value=data.edge_attr,  # start from 1 to 4
                         sparse_sizes=(N, N)).coalesce().to_dense()
        edge_attr = A.reshape(-1, 1)
    edge_attr = edge_attr.to(data.edge_attr.dtype)
    data.edge_attr = edge_attr
    adj = torch.ones([N, N], dtype=torch.long)
    edge_index = dense_to_sparse(adj)[0]
    data.edge_index = edge_index

    # pestat_node = []
    # pestat_edge = []
    # for pename in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'rrwp']:
    #     pestat_var = f"pestat_{pename}"
    #     if data.get(pestat_var, None) is not None:
    #         pestat_node.append(getattr(data, pestat_var))
    # if data.get('rrwp', None) is not None:
    #     pestat_node.append(data.rrwp)
    # if pestat_node:
    #     pestat_node = torch.cat(pestat_node, dim=-1)
    #     data.pestat_node = pestat_node
    #
    # for pename in ['rrwp_edge', 'RD_val']:
    #     if data.get(pename, None) is not None:
    #         pestat_edge.append(getattr(data, pename))
    # if pestat_edge:
    #     pestat_edge = torch.cat(pestat_edge, dim=-1)
    #     # pestat_edge_padded = torch.zeros([(N+1), (N+1), pestat_edge.shape[-1]], dtype=torch.float)
    #     # pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
    #     data.pestat_edge = pestat_edge.reshape(N*N, -1)

    data.num_nodes = N
    return data


def pretransform_pe(data, add_virtual_node=False):
    pestat_node = []
    pestat_edge = []
    N = data.num_nodes - 1 if add_virtual_node else data.num_nodes
    for pename in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'rrwp']:
        pestat_var = f"pestat_{pename}"
        if data.get(pestat_var, None) is not None:
            pestat_node.append(getattr(data, pestat_var))
    if data.get('rrwp', None) is not None:
        pestat_node.append(data.rrwp)
    if pestat_node:
        pestat_node = torch.cat(pestat_node, dim=-1)
        data.pestat_node = pestat_node
    if add_virtual_node:
        if pestat_node:
            data.pestat_node = torch.cat([data.pestat_node, torch.zeros([1, pestat_node.shape[1]], dtype=torch.float)], dim=0)

        for pename in ['rrwp_edge']:
            if data.get(pename, None) is not None:
                pestat_edge.append(getattr(data, pename))
        if pestat_edge:
            pestat_edge = torch.cat(pestat_edge, dim=-1)
            pestat_edge_padded = torch.zeros([(N + 1), (N + 1), pestat_edge.shape[-1]], dtype=torch.float)
            pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
            data.pestat_edge = pestat_edge_padded.reshape((N + 1) * (N + 1), -1)
    else:
        for pename in ['rrwp_edge']:
            if data.get(pename, None) is not None:
                pestat_edge.append(getattr(data, pename))
        if pestat_edge:
            pestat_edge = torch.cat(pestat_edge, dim=-1)
            # pestat_edge_padded = torch.zeros([(N+1), (N+1), pestat_edge.shape[-1]], dtype=torch.float)
            # pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
            data.pestat_edge = pestat_edge.reshape(N * N, -1)
            if data.get('RD_matrix', None) is not None:
                data.pestat_edge = torch.cat([data.pestat_edge, data.RD_matrix.reshape(N * N, 1)], dim=-1)
        elif data.get('RD_matrix', None) is not None:
            data.pestat_edge = data.RD_matrix.reshape(N * N, 1)
    return data


def pretransform_QM9_generation(data, target=None, align=False, add_virtual_node=False, mean=None, std=None):
    N = data.num_nodes if not add_virtual_node else data.num_nodes + 1
    # data.num_node_per_graph = torch.tensor((N), dtype=torch.long)

    data.x_original = data.x.clone()  # 5 atom types
    # data.edge_index_original = data.edge_index.clone()
    # data.edge_attr_original = data.edge_attr.clone()
    # adj = SparseTensor(row=data.edge_index[0],
    #                    col=data.edge_index[1],
    #                    value=torch.ones(data.edge_index.shape[1]),
    #                    sparse_sizes=(N, N)).coalesce().to_dense()
    # data.original_edge = adj.reshape(-1).bool()
    if align:
        data.x_complex = data.z.clone().detach().unsqueeze(1)
        x_simplified = data.z.clone().detach()
        x_simplified[x_simplified == 1] = 0
        x_simplified[x_simplified == 6] = 1
        x_simplified[x_simplified == 7] = 2
        x_simplified[x_simplified == 8] = 3
        x_simplified[x_simplified == 9] = 4
        data.x = x_simplified.unsqueeze(1)
        data.x_simplified = x_simplified
        data.edge_attr = (torch.nonzero(data.edge_attr)[:, 1] + 1).long()
    # print(data.x, data.edge_attr)
    # A = SparseTensor(row=data.edge_index[0],
    #                  col=data.edge_index[1],
    #                  value=data.edge_attr,  # start from 1 to 4
    #                  sparse_sizes=(N, N)).coalesce().to_dense()
    # edge_attr = A.reshape(-1, 1).long()
    # data.edge_attr = edge_attr
    # adj = torch.ones([N, N], dtype=torch.long)
    # edge_index = dense_to_sparse(adj)[0]
    # data.edge_index = edge_index
    if target is not None:
        data.y = data.y[0, int(target)].unsqueeze(0)
        data.y_mean = mean[int(target)].unsqueeze(0)
        data.y_std = std[int(target)].unsqueeze(0)

    # pestat_node = []
    # pestat_edge = []
    # if add_virtual_node:
    #     data.x = torch.cat(
    #         [data.x, torch.ones([1, 1], dtype=torch.long) * (cfg.dataset.node_encoder_num_types + 1)], dim=0)
    #
    #     for pename in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'rrwp']:
    #         pestat_var = f"pestat_{pename}"
    #         if data.get(pestat_var, None) is not None:
    #             pestat_node.append(getattr(data, pestat_var))
    #     if data.get('rrwp', None) is not None:
    #         pestat_node.append(data.rrwp)
    #     if pestat_node:
    #         pestat_node = torch.cat(pestat_node, dim=-1)
    #         pestat_node = torch.cat([pestat_node, torch.zeros([1, pestat_node.shape[1]], dtype=torch.float)], dim=0)
    #         data.pestat_node = pestat_node
    #
    #     for pename in ['rrwp_edge', 'RD_val']:
    #         if data.get(pename, None) is not None:
    #             pestat_edge.append(getattr(data, pename))
    #     if pestat_edge:
    #         pestat_edge = torch.cat(pestat_edge, dim=-1)
    #         pestat_edge_padded = torch.zeros([(N + 1), (N + 1), pestat_edge.shape[-1]], dtype=torch.float)
    #         pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
    #         data.pestat_edge = pestat_edge_padded.reshape((N + 1) * (N + 1), -1)
    # else:
    #     for pename in ['rrwp_edge', 'RD_val']:
    #         if data.get(pename, None) is not None:
    #             pestat_edge.append(getattr(data, pename))
    #     if pestat_edge:
    #         pestat_edge = torch.cat(pestat_edge, dim=-1)
    #         # pestat_edge_padded = torch.zeros([(N+1), (N+1), pestat_edge.shape[-1]], dtype=torch.float)
    #         # pestat_edge_padded[:N, :N] = pestat_edge.reshape(N, N, -1)
    #         data.pestat_edge = pestat_edge.reshape(N * N, -1)

    # if data.get('log_deg', None) is not None:
    #     data.log_deg = torch.cat([data.log_deg, torch.log(torch.tensor([N], dtype=torch.float))], dim=0)
    # if data.get('deg', None) is not None:
    #     data.log_deg = torch.cat([data.deg, torch.tensor([N], dtype=torch.long)], dim=0)

    #
    # if data.get('log_deg', None) is not None:
    #     data.log_deg = torch.cat([data.log_deg, torch.log(torch.tensor([N], dtype=torch.float))], dim=0)
    # if data.get('deg', None) is not None:
    #     data.log_deg = torch.cat([data.deg, torch.tensor([N], dtype=torch.long)], dim=0)
    return data
