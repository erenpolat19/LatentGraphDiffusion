import math
import logging
import numpy as np
from typing import Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym.register import act_dict, register_layer
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
import warnings
from .utils import *
from lgd.model.GraphTransformerEncoder import *
# from watchpoints import watch
# watch.config(pdb=True)

# TODO: the first embedding layer of unknown nodes/edges to be predicted, use same embedding?
# TODO: what about the local attention between nodes connected with unknown edges? (especially in difusion model)
# TODO: global attention, do we need linear approximation?
# TODO: initialization of virtual node (also graph embedding)? how many virtual nodes?
# TODO: positional and structural encoding? also degree information? \
#  (see ablation study in Graph Inductive Biases in Transformers without Message Passing)
# TODO: data augmentation and contrastive learning? will only use global features for contrastive loss
# TODO: softmax cannot simulate sum over neighboring nodes in message passing



@register_network('GraphTransformerSyntheticEncoder')
class GraphTransformerSyntheticEncoder(nn.Module):
    """
        Full Graph Transformer Encoder; encode structural information by four types of reconstruction losses
    """
    def __init__(self, dim_in=0, dim_out=0, # in_dim, posenc_in_dim, posenc_dim, prefix_dim, hid_dim, out_dim,
    #              use_time, temb_dim, num_heads, num_layers, pool,
    #              dropout=0.0,
    #              attn_dropout=0.0,
    #              layer_norm=False, batch_norm=True,
    #              residual=True,
    #              act='relu',
    #              norm_e=True,
    #              O_e=True,
    #              ff_e=False,
                 cfg=None, # need to explicitly enter cfg.encoder or cfg.cond
                 **kwargs):
        super().__init__()

        # if cfg is None:
        #     cfg = dict()
        # logging.info(cfg)
        # cfg = cfg.encoder
        self.in_dim = cfg.in_dim
        self.prefix_dim = cfg.prefix_dim
        self.posenc_in_dim = cfg.posenc_in_dim
        self.posenc_in_dim_edge = cfg.posenc_in_dim_edge
        self.posenc_dim = cfg.posenc_dim
        self.hid_dim = cfg.hid_dim
        self.out_dim = cfg.out_dim
        self.decode_dim = cfg.get("decode_dim", 1)
        self.use_time = cfg.use_time
        self.temb_dim = cfg.temb_dim
        self.num_heads = cfg.num_heads
        self.num_layers = cfg.num_layers
        self.dropout = cfg.dropout
        self.attn_dropout = cfg.attn_dropout
        self.residual = cfg.residual
        self.layer_norm = cfg.layer_norm
        self.batch_norm = cfg.batch_norm
        self.act = act_dict[cfg.act]() if cfg.act is not None else nn.Identity()
        self.update_e = cfg.get("update_e", True)
        self.force_undirected = cfg.get("force_undirected", False)
        self.attn_product = cfg.attn.get("attn_product", 'mul'),
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner

        self.node_dict_dim = cfg.get('node_encoder_num_types', 1)   # TODO: update this dimension of dictionary
        self.edge_dict_dim = cfg.get('edge_encoder_num_types', 2)  # whether there's an edge or not
        if cfg.get('add_virtual_node_edge', True):
            self.node_dict_dim = self.node_dict_dim + 10  # 2
            self.edge_dict_dim = self.edge_dict_dim + 10  # 5
        # self.node_emb = nn.Embedding(self.node_dict_dim, self.in_dim, padding_idx=0)
        # self.edge_emb = nn.Embedding(self.edge_dict_dim, self.in_dim, padding_idx=0)
        if cfg.node_encoder_name == 'Embedding':
            self.node_emb = nn.Embedding(self.node_dict_dim, self.in_dim, padding_idx=0)
        else:
            NodeEncoder = register.node_encoder_dict[cfg.node_encoder_name]
            self.node_emb = NodeEncoder(self.in_dim)
        if cfg.edge_encoder_name == 'Embedding':
            self.edge_emb = nn.Embedding(self.edge_dict_dim, self.in_dim, padding_idx=0)
        else:
            EdgeEncoder = register.edge_encoder_dict[cfg.edge_encoder_name]
            self.edge_emb = EdgeEncoder(self.in_dim)

        self.num_tasks = cfg.get("num_tasks", 1)
        self.prefix_type = cfg.get("prefix_type", "add_virtual")
        assert self.prefix_type in ["add_virtual", "add_all", "concat"]
        if self.prefix_type == "concat":
            assert cfg.in_dim + cfg.prefix_dim + cfg.posenc_dim == cfg.hid_dim
        else:
            assert cfg.prefix_dim == cfg.hid_dim
            assert cfg.in_dim + cfg.posenc_dim == cfg.hid_dim
        self.prefix_emb = nn.Embedding(self.num_tasks, self.prefix_dim, padding_idx=0)  # TODO: whether need this padding_idx=0?

        self.task_type = cfg.get('task_type', 'regression')
        if cfg.get('label_raw_norm', None) == 'BatchNorm':
            label_norm = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
        elif cfg.get('label_raw_norm', None) == 'LayerNorm':
            label_norm = nn.LayerNorm(self.hid_dim)
        else:
            label_norm = nn.Identity()
        self.label_embed_regression = nn.Sequential(nn.Linear(1, 2 * self.hid_dim), self.act,
                                                    nn.Linear(2 * self.hid_dim, self.hid_dim), label_norm, self.act)
        self.pseudo_label = nn.Parameter(torch.zeros(1, self.hid_dim))
        self.label_embed_classification = nn.Embedding(cfg.get("num_classes", 2) + 1, self.hid_dim)
        self.label_embed_type = cfg.get('label_embed_type', 'add_virtual')
        assert self.label_embed_type in ['add_virtual', 'add_all']

        if self.posenc_in_dim > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_node = nn.BatchNorm1d(self.posenc_in_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_node = nn.LayerNorm(self.posenc_in_dim)
            else:
                pe_raw_norm_node = nn.Identity()
            self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, self.posenc_dim))
            # self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, 2 * self.posenc_dim), self.act,
            #                                 nn.Linear(2 * self.posenc_dim, self.posenc_dim))
        if self.posenc_in_dim_edge > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_edge = nn.BatchNorm1d(self.posenc_in_dim_edge, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_edge = nn.LayerNorm(self.posenc_in_dim_edge)
            else:
                pe_raw_norm_edge = nn.Identity()
            # self.posenc_emb_edge = nn.Sequential(pe_raw_norm_edge, nn.Linear(self.posenc_in_dim_edge, 2 * self.posenc_dim), self.act,
            #                                      nn.Linear(2 * self.posenc_dim, self.posenc_dim))
            self.posenc_emb_edge = nn.Sequential(pe_raw_norm_edge, nn.Linear(self.posenc_in_dim_edge, self.posenc_dim))

        self.node_in_mlp = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.edge_in_mlp = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        if self.layer_norm:
            self.layer_norm_in_h = nn.LayerNorm(self.hid_dim)
            self.layer_norm_in_e = nn.LayerNorm(self.hid_dim) if cfg.norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum) if cfg.norm_e else nn.Identity()

        self.temb_layer = nn.Sequential(nn.Linear(self.temb_dim, 2 * self.temb_dim), self.act,
                                        nn.Linear(2 * self.temb_dim, self.temb_dim)) if self.temb_dim > 0 else None

        self.GTE_layers = nn.ModuleList([
            GraphTransformerEncoderLayer(in_dim=self.hid_dim,
                                         out_dim=self.hid_dim,
                                         temb_dim=self.temb_dim,
                                         num_heads=self.num_heads,
                                         dropout=self.dropout,
                                         attn_dropout=self.attn_dropout,
                                         layer_norm=self.layer_norm, batch_norm=self.batch_norm,
                                         residual=self.residual,
                                         act=cfg.act,
                                         norm_e=cfg.norm_e,
                                         O_e=cfg.O_e,
                                         ff_e=cfg.ff_e,
                                         cfg=cfg) for _ in range(self.num_layers)
        ])

        self.final_layer_node = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_layer_edge = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_norm = cfg.get('final_norm', False)
        if self.final_norm:
            self.final_norm_node_1 = nn.LayerNorm(self.out_dim)
            self.final_norm_edge_1 = nn.LayerNorm(self.out_dim)
        assert cfg.pool in ['max', 'add', 'mean', 'none']
        self.pool = cfg.pool
        self.pool_vn = cfg.get('pool_vn', False)
        self.pool_edge = cfg.get('pool_edge', False)
        self.post_pool = cfg.get('post_pool', False)
        if self.pool != 'none':
            self.global_pool = eval("global_" + self.pool + "_pool")
            self.graph_out_mlp = nn.Sequential(nn.Linear(self.out_dim, 2 * self.out_dim), self.act,
                                               nn.Linear(2 * self.out_dim, self.out_dim)) if self.post_pool else nn.Identity()
            if self.final_norm:
                self.final_norm_node_2 = nn.LayerNorm(self.out_dim)
            if self.pool_edge:
                self.graph_out_mlp_2 = nn.Sequential(nn.Linear(self.out_dim, 2 * self.out_dim), self.act,
                                                     nn.Linear(2 * self.out_dim, self.out_dim)) if self.post_pool else nn.Identity()
                if self.final_norm:
                    self.final_norm_edge_2 = nn.LayerNorm(self.out_dim)
        self.decode_node = nn.Linear(self.out_dim, self.node_dict_dim)  # virtual node
        self.decode_edge = nn.Linear(self.out_dim, self.edge_dict_dim)  # no edge, virtual edge, loop
        self.decode_graph = nn.Linear(self.out_dim, self.decode_dim)
        # self.decode_node_from_edge = nn.Linear(self.out_dim, self.node_dict_dim)  # virtual node
        self.decode_edge_from_node = nn.Linear(self.out_dim, self.edge_dict_dim)
        self.decode_edge_from_tuple = nn.Linear(self.out_dim, self.edge_dict_dim)

    def forward(self, batch, t=None, prefix=None, label=None, **kwargs):
        # batch.x_0 = batch.x
        # batch.edge_attr_0 = batch.edge_attr
        num_nodes, num_edges = batch.num_nodes, batch.edge_index.shape[1]
        batch_num_node = batch.num_node_per_graph
        batch_node_idx = num2batch(batch_num_node)
        assert torch.equal(batch_node_idx, batch.batch)
        batch_edge_idx = num2batch(batch_num_node ** 2)
        virtual_node_idx = torch.cumsum(batch_num_node, dim=0) - 1
        h = self.node_emb(batch.x).reshape(num_nodes, -1)
        e = self.edge_emb(batch.edge_attr).reshape(num_edges, -1)

        if self.posenc_dim > 0:
            if self.posenc_in_dim > 0 and batch.get("pestat_node", None) is not None:
                batch_posenc_emb = self.posenc_emb(batch.pestat_node)
            else:
                batch_posenc_emb = torch.zeros([num_nodes, self.posenc_dim], dtype=torch.float, device=batch.x.device)
            if self.posenc_in_dim_edge > 0 and batch.get("pestat_edge", None) is not None:
                batch_posenc_emb_edge = self.posenc_emb_edge(batch.pestat_edge)
            else:
                batch_posenc_emb_edge = torch.zeros([num_edges, self.posenc_dim], dtype=torch.float, device=batch.x.device)
            # h = torch.cat([h, prefix_h, batch_posenc_emb], dim=1)
            # e = torch.cat([e, prefix_e, batch_posenc_edge], dim=1)
            h = torch.cat([h, batch_posenc_emb], dim=1)
            e = torch.cat([e, batch_posenc_emb_edge], dim=1)
        # else:
        #     h = torch.cat([h, prefix_h], dim=1)
        #     e = torch.cat([e, prefix_e], dim=1)

        if batch.get("prefix", None) is not None:
            prefix = batch.prefix
        if prefix is not None:
            if prefix.dtype == torch.long:
                prefix = self.prefix_emb(prefix)  # this is not encouraged, as we want to fix the parameters of encoder while prefix-tuning
            else:
                assert len(prefix.shape) == 1 and prefix.shape[0] == self.prefix_dim
            if self.prefix_type == "concat":  # by default concat to all nodes and edge vectors
                prefix_h = prefix.unsqueeze(0).repeat(num_nodes, 1)
                prefix_e = prefix.unsqueeze(0).repeat(num_edges, 1)
                h = torch.cat([h, prefix_h], dim=1)
                e = torch.cat([e, prefix_e], dim=1)
            elif self.prefix_type == 'add_virtual':
                h[virtual_node_idx] = h[virtual_node_idx] + prefix.unsqueeze(0).repeat(batch.num_graphs, 1)
            else:
                prefix_h = prefix.unsqueeze(0).repeat(num_nodes, 1)
                prefix_e = prefix.unsqueeze(0).repeat(num_edges, 1)
                h = h + prefix_h
                e = e + prefix_e
        # else:
        #     prefix = torch.zeros([self.prefix_dim], dtype=torch.float, device=batch.x.device)

        if label is not None:
            if not torch.is_tensor(label):
                label, masked_label_idx = label
            else:
                masked_label_idx = None
            if len(label.shape) == 1:  # TODO: check this for molpcba
                label = label.unsqueeze(1)
            if self.task_type == 'regression':
                label = self.label_embed_regression(label)
            else:
                label = self.label_embed_classification(label)  # do not encourage this while transfering
                if len(label.shape) == 3:
                    label = label.squeeze(1)
            # else:
            #     assert label.shape[1] == self.hid_dim  # embedding labels outside the models is encouraged while transfering
            if masked_label_idx is not None:
                label = label.clone()
                label[masked_label_idx] = 0

            if self.label_embed_type == 'add_virtual':
                h[virtual_node_idx] = h[virtual_node_idx] + label
            else:
                label_embed_node = label[batch_node_idx]
                label_embed_edge = label[batch_edge_idx]
                h = h + label_embed_node
                e = e + label_embed_edge
        # else:
        #     label = self.pseudo_label

        h = self.node_in_mlp(h)
        e = self.edge_in_mlp(e)
        if self.layer_norm:
            h = self.layer_norm_in_h(h)
            e = self.layer_norm_in_e(e)
        if self.batch_norm:
            h = self.batch_norm_in_h(h)
            e = self.batch_norm_in_e(e)

        batch.x = h
        batch.edge_attr = e

        if self.use_time and t is not None:
            temb = get_timestep_embedding(t, self.temb_dim)
            temb = self.temb_layer(temb)
            temb_h, temb_e = temb[batch_node_idx], temb[batch_edge_idx]
            temb = (temb_h, temb_e)
        else:
            temb = None

        for _ in range(self.num_layers):
            batch = self.GTE_layers[_](batch, temb)

        batch.x = self.final_layer_node(batch.x)
        if self.force_undirected:
            A = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
            A = (A + A.permute(0, 2, 1, 3)).reshape(-1, A.shape[-1])
            mask = A.any(dim=1)
            batch.edge_attr = A[mask]
            assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
        batch.edge_attr = self.final_layer_edge(batch.edge_attr)
        if self.final_norm:
            batch.x = self.final_norm_node_1(batch.x)
            batch.edge_attr = self.final_norm_edge_1(batch.edge_attr)

        if self.pool != 'none':
            v_g = self.graph_out_mlp(self.global_pool(batch.x, batch_node_idx))
            if self.final_norm:
                v_g = self.final_norm_node_2(v_g)
            if self.pool_vn:
                v_g = v_g + batch.x[virtual_node_idx]
            if self.pool_edge:
                v_e = self.graph_out_mlp_2(self.global_pool(batch.edge_attr, batch_edge_idx))
                if self.final_norm:
                    v_e = self.final_norm_edge_2(v_e)
                v_g = v_g + v_e
            # TODO: do we need to change the virtual nodes in batch.x? used for classification loss
            # batch.x[virtual_node_idx] = v_g
        else:
            v_g = batch.x[virtual_node_idx]
        batch.graph_attr = v_g
        return batch

    def encode(self, batch, t=None, prefix=None, label=None, **kwargs):
        return self.forward(batch, t, prefix, label, **kwargs)

    def decode(self, batch, **kwargs):
        # TODO: implement decode
        return self.decode_node(batch.x), self.decode_edge(batch.edge_attr), self.decode_graph(batch.graph_attr).flatten()

    def decode_recon(self, batch, **kwargs):
        src = batch.x[batch.edge_index[0]]  # (num_nodes) x num_heads x out_dim
        dest = batch.x[batch.edge_index[1]]  # (num_nodes) x num_heads x out_dim
        score_edge = torch.mul(src, dest) if self.attn_product == 'mul' else (src + dest)  # element-wise multiplication;

        score_tuple = torch.mul(score_edge, batch.edge_attr)

        return self.decode_edge_from_node(score_edge), self.decode_edge_from_tuple(score_tuple)

    def encode_label(self, label, **kwargs):
        if len(label.shape) == 1:
            label = label.unsqueeze(1)
        if self.task_type == 'regression':
            label = self.label_embed_regression(label)
        else:
            label = self.label_embed_classification(label)  # do not encourage this while transfering
            if len(label.shape) == 3:
                label = label.squeeze(1)
        return label


