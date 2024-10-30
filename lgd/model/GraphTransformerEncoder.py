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


# from watchpoints import watch
# watch.config(pdb=True)


class GINE(nn.Module):
    """
        Self-attention in graph transformer encoder
    """

    def __init__(self, in_dim, out_dim,
                 dropout=0., act=None,
                 edge_enhance=True,
                 project_edge=True,
                 edge_dim: Optional[int] = None,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        self.edge_enhance = edge_enhance
        if edge_enhance:
            if edge_dim is None:
                edge_dim = in_dim
            if project_edge:
                self.lin_edge = nn.Linear(edge_dim, in_dim)
            else:
                assert edge_dim == in_dim
                self.lin_edge = nn.Identity()
        else:
            self.lin_edge = None

        self.mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.out_dim), self.act,
                                 nn.Linear(2 * self.out_dim, self.out_dim))

    def forward(self, batch):
        if batch.get('original_edge', None) is not None:
            real_edge = batch.original_edge
            edge_idx = batch.edge_index[:, real_edge]
        else:
            edge_idx = batch.get('edge_index_original', batch.edge_index)
            if edge_idx.shape[1] != batch.edge_index.shape[1]:
                raise NotImplementedError  # do not process
            real_edge = torch.ones(edge_idx.shape[1], dtype=torch.long, device=batch.x.device).bool()

        h = batch.x
        e = batch.edge_attr[real_edge]

        src = h[edge_idx[0]]
        msg = self.act(src + self.lin_edge(e)) if self.edge_enhance else self.act(src)
        aggr = torch.zeros_like(h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_idx[1], dim=0, out=aggr, reduce='add')

        h_out = self.mlp(h + self.dropout(aggr))
        return h_out


class GTE_Attention(nn.Module):
    """
        Self-attention in graph transformer encoder
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 score_act=True,
                 signed_sqrt=False,
                 scaled_attn=True,
                 attn_product='mul',
                 attn_reweight=False,
                 edge_reweight=False,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance
        self.score_act = score_act
        self.signed_sqrt = signed_sqrt
        self.scaled_attn = scaled_attn
        assert attn_product in ['add', 'mul']
        self.attn_product = attn_product
        self.attn_reweight = attn_reweight
        self.edge_reweight = edge_reweight

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        if self.attn_reweight:
            self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
            nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance and self.edge_reweight:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        # TODO: preprocess a connected graph for nodes (including virtual nodes) in the same graph,\
        #  recorded as batch.edge_index; as for original edge_index/attr, maintain as edge_index_original/edge_attr_original
        src = batch.K_h[batch.edge_index[0]]  # (num_nodes) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]  # (num_nodes) x num_heads x out_dim
        score = torch.mul(src, dest) if self.attn_product == 'mul' else (src + dest)  # element-wise multiplication;
        # isnan = torch.isnan(score).any()
        # if isnan:
        #     logging.info("nan, 1")
        #     score[torch.isnan(score)] = 0
        # TODO: some other solutions including add or global attention; also consider symmetry

        # if batch.get("E", None) is not None
        batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
        # (num node) x num_heads x out_dim
        score = score * E_w

        if self.signed_sqrt:
            score = torch.sqrt(torch.relu(score).clamp_min(1e-8)) - torch.sqrt(
                torch.relu(-score).clamp_min(1e-8))  # TODO: whether this is necessary
        score = score + E_b

        if self.score_act:
            score = self.act(score)
        e_t = score

        # output edge
        # if batch.get("E", None) is not None:
        batch.wE = score.flatten(1)

        # final attention
        score = torch.einsum("ehd, dhc->ehc", score, self.Aw) if self.attn_reweight else score.sum(-1, keepdims=True)

        if self.scaled_attn:
            score = score / torch.sqrt(torch.tensor([self.out_dim], dtype=torch.float, device=batch.x.device))
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # raw_attn = score
        score = pyg_softmax(score, batch.edge_index[1], batch.num_nodes)  # (num node) x num_heads x 1

        # TODO: check this num_nodes, should be total number of nodes in the batch
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num node) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance:
            rowV = torch.einsum("nhd, dhc -> nhc", e_t, self.VeRow) if self.edge_reweight else e_t
            rowV = scatter(rowV * score, batch.edge_index[1], dim=0, reduce="add")
            # rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            # rowV = torch.einsum("nhd, dhc -> nhc", rowV, self.VeRow)
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:  # remove this if afterwards
            batch.E = self.E(batch.edge_attr)
        else:
            raise NotImplementedError  # this should be processed in the encoder part for unattributed edges

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


@register_layer("GTE_layer")
class GraphTransformerEncoderLayer(nn.Module):
    """
        Full Transformer Layer of Graph Transformer Encoder
    """

    def __init__(self, in_dim, out_dim, temb_dim, num_heads, final_dim=None,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 ff_e=False,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.final_dim = out_dim if final_dim is None else final_dim
        self.temb_dim = temb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = cfg.get("update_e", True)
        self.force_undirected = cfg.get("force_undirected", False)
        self.bn_momentum = cfg.attn.bn_momentum
        self.bn_no_runner = cfg.attn.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", False)

        self.attention = GTE_Attention(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            score_act=cfg.attn.get("score_act", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn=cfg.attn.get("scaled_attn", True),
            no_qk=cfg.attn.get("no_qk", False),
            attn_product=cfg.attn.get("attn_product", 'mul'),
            attn_reweight=cfg.attn.get("attn_reweight", False),
            edge_reweight=cfg.attn.get("edge_reweight", False)
        )

        # if cfg.attn.get('graphormer_attn', False):
        #     self.attention = MultiHeadAttentionLayerGraphormerSparse(
        #         in_dim=in_dim,
        #         out_dim=out_dim // num_heads,
        #         num_heads=num_heads,
        #         use_bias=cfg.attn.get("use_bias", False),
        #         dropout=attn_dropout,
        #         clamp=cfg.attn.get("clamp", 5.),
        #         act=cfg.attn.get("act", "relu"),
        #         edge_enhance=True,
        #         score_act=cfg.attn.get("score_act", False),
        #         signed_sqrt=cfg.attn.get("signed_sqrt", False),
        #         scaled_attn =cfg.attn.get("scaled_attn", False),
        #         no_qk=cfg.attn.get("no_qk", False),
        #     )

        if cfg.get('mpnn', None) is not None:
            self.message_passing = cfg.mpnn.get('enable', True)
        else:
            self.message_passing = False
        if self.message_passing:
            self.mpnn = GINE(in_dim, out_dim, dropout=cfg.mpnn.get('dropout', 0.0),
                             act=cfg.mpnn.get('act', 'relu'), edge_enhance=cfg.mpnn.get('edge_enhance', True),
                             project_edge=cfg.mpnn.get('project_edge', False), edge_dim=in_dim)

        self.temb_proj_h = nn.Linear(temb_dim, out_dim)
        self.temb_proj_e = nn.Linear(temb_dim, out_dim)

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()
        self.ff_e = ff_e

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, self.final_dim)

        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, self.final_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(self.final_dim)
            self.layer_norm2_e = nn.LayerNorm(self.final_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(self.final_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=cfg.bn_momentum)
            self.batch_norm2_e = nn.BatchNorm1d(self.final_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch, temb=None):
        h = batch.x
        num_nodes = batch.num_nodes

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None

        if temb is not None:
            # TODO: temb should be align with size of node and edge features; process it in the higher-level model
            temb_h, temb_e = temb
            temb_h = self.temb_proj_h(self.act(temb_h))
            temb_e = self.temb_proj_e(self.act(temb_e))
            batch.x = batch.x + temb_h
            batch.edge_attr = batch.edge_attr + temb_e

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            log_deg = get_log_deg(batch)
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.message_passing:
            h_mp = self.mpnn(batch)
            h = h + h_mp

        if self.residual and self.in_dim == self.out_dim:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual and self.out_dim == self.final_dim:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        if self.ff_e:
            e_in2 = e
            e = self.FFN_e_layer1(e)
            e = self.act(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer2(e)

            if self.residual and self.out_dim == self.final_dim:
                e = e_in2 + e
            if self.layer_norm:
                e = self.layer_norm2_e(e)
            if self.batch_norm:
                e = self.batch_norm2_e(e)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@register_network('GraphTransformerEncoder')
class GraphTransformerEncoder(nn.Module):
    """
        Full Graph Transformer Encoder
    """

    def __init__(self, dim_in=0, dim_out=0,  # in_dim, posenc_in_dim, posenc_dim, prefix_dim, hid_dim, out_dim,
                 #              use_time, temb_dim, num_heads, num_layers, pool,
                 #              dropout=0.0,
                 #              attn_dropout=0.0,
                 #              layer_norm=False, batch_norm=True,
                 #              residual=True,
                 #              act='relu',
                 #              norm_e=True,
                 #              O_e=True,
                 #              ff_e=False,
                 cfg=None,  # need to explicitly enter cfg.encoder or cfg.cond
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
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner

        self.node_dict_dim = cfg.get('node_encoder_num_types', 1)  # TODO: update this dimension of dictionary
        self.edge_dict_dim = cfg.get('edge_encoder_num_types', 1)
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
        self.prefix_emb = nn.Embedding(self.num_tasks, self.prefix_dim,
                                       padding_idx=0)  # TODO: whether need this padding_idx=0?

        self.task_type = cfg.get('task_type', 'regression')
        if cfg.get('label_raw_norm', None) == 'BatchNorm':
            label_norm = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                        momentum=self.bn_momentum)
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
                pe_raw_norm_node = nn.BatchNorm1d(self.posenc_in_dim, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_node = nn.LayerNorm(self.posenc_in_dim)
            else:
                pe_raw_norm_node = nn.Identity()
            self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, self.posenc_dim))
            # self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, 2 * self.posenc_dim), self.act,
            #                                 nn.Linear(2 * self.posenc_dim, self.posenc_dim))
        if self.posenc_in_dim_edge > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_edge = nn.BatchNorm1d(self.posenc_in_dim_edge, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
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
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum) if cfg.norm_e else nn.Identity()

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
                                               nn.Linear(2 * self.out_dim,
                                                         self.out_dim)) if self.post_pool else nn.Identity()
            if self.final_norm:
                self.final_norm_node_2 = nn.LayerNorm(self.out_dim)
            if self.pool_edge:
                self.graph_out_mlp_2 = nn.Sequential(nn.Linear(self.out_dim, 2 * self.out_dim), self.act,
                                                     nn.Linear(2 * self.out_dim,
                                                               self.out_dim)) if self.post_pool else nn.Identity()
                if self.final_norm:
                    self.final_norm_edge_2 = nn.LayerNorm(self.out_dim)
        self.decode_node = nn.Linear(self.out_dim, self.node_dict_dim)  # virtual node
        self.decode_edge = nn.Linear(self.out_dim, self.edge_dict_dim)  # no edge, virtual edge, loop
        self.decode_graph = nn.Linear(self.out_dim, self.decode_dim)

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
                batch_posenc_emb_edge = torch.zeros([num_edges, self.posenc_dim], dtype=torch.float,
                                                    device=batch.x.device)
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
                prefix = self.prefix_emb(
                    prefix)  # this is not encouraged, as we want to fix the parameters of encoder while prefix-tuning
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
        return self.decode_node(batch.x), self.decode_edge(batch.edge_attr), self.decode_graph(
            batch.graph_attr).flatten()

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


# TODO: in training, register the prompt vector into model and notice the optimizer; storage of the prompt vector


@register_network('GraphTransformerDecoder')
class GraphTransformerDecoder(nn.Module):
    """
        Full Graph Transformer Encoder/Decoder; freeze encoder
    """

    def __init__(self, dim_in=0, dim_out=0,  # in_dim, posenc_in_dim, posenc_dim, prefix_dim, hid_dim, out_dim,
                 #              use_time, temb_dim, num_heads, num_layers, pool,
                 #              dropout=0.0,
                 #              attn_dropout=0.0,
                 #              layer_norm=False, batch_norm=True,
                 #              residual=True,
                 #              act='relu',
                 #              norm_e=True,
                 #              O_e=True,
                 #              ff_e=False,
                 cfg=None,  # need to explicitly enter cfg.encoder or cfg.cond
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
        self.num_decode_layers = cfg.num_decode_layers
        self.dropout = cfg.dropout
        self.attn_dropout = cfg.attn_dropout
        self.residual = cfg.residual
        self.layer_norm = cfg.layer_norm
        self.batch_norm = cfg.batch_norm
        self.act = act_dict[cfg.act]() if cfg.act is not None else nn.Identity()
        self.update_e = cfg.get("update_e", True)
        self.force_undirected = cfg.get("force_undirected", False)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner

        self.node_dict_dim = cfg.get('node_encoder_num_types', 1)  # TODO: update this dimension of dictionary
        self.edge_dict_dim = cfg.get('edge_encoder_num_types', 1)
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
        self.prefix_emb = nn.Embedding(self.num_tasks, self.prefix_dim,
                                       padding_idx=0)  # TODO: whether need this padding_idx=0?

        self.task_type = cfg.get('task_type', 'regression')
        if cfg.get('label_raw_norm', None) == 'BatchNorm':
            label_norm = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                        momentum=self.bn_momentum)
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
                pe_raw_norm_node = nn.BatchNorm1d(self.posenc_in_dim, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_node = nn.LayerNorm(self.posenc_in_dim)
            else:
                pe_raw_norm_node = nn.Identity()
            self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, self.posenc_dim))
            # self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, 2 * self.posenc_dim), self.act,
            #                                 nn.Linear(2 * self.posenc_dim, self.posenc_dim))
        if self.posenc_in_dim_edge > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_edge = nn.BatchNorm1d(self.posenc_in_dim_edge, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
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
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum) if cfg.norm_e else nn.Identity()

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
                                               nn.Linear(2 * self.out_dim,
                                                         self.out_dim)) if self.post_pool else nn.Identity()
            if self.final_norm:
                self.final_norm_node_2 = nn.LayerNorm(self.out_dim)
            if self.pool_edge:
                self.graph_out_mlp_2 = nn.Sequential(nn.Linear(self.out_dim, 2 * self.out_dim), self.act,
                                                     nn.Linear(2 * self.out_dim,
                                                               self.out_dim)) if self.post_pool else nn.Identity()
                if self.final_norm:
                    self.final_norm_edge_2 = nn.LayerNorm(self.out_dim)

        self.decode_node_emb = nn.Linear(self.out_dim, self.hid_dim)
        self.decode_edge_emb = nn.Linear(self.out_dim, self.hid_dim)
        self.decode_graph_emb = nn.Linear(self.out_dim, self.hid_dim)
        self.decode_layers = nn.ModuleList([
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
                                         cfg=cfg) for _ in range(self.num_decode_layers)
        ])

        self.decode_node = nn.Linear(self.hid_dim, self.node_dict_dim)  # virtual node
        self.decode_edge = nn.Linear(self.hid_dim, self.edge_dict_dim)  # no edge, virtual edge, loop
        self.decode_graph = nn.Linear(self.hid_dim, self.decode_dim)

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
                batch_posenc_emb_edge = torch.zeros([num_edges, self.posenc_dim], dtype=torch.float,
                                                    device=batch.x.device)
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
                prefix = self.prefix_emb(
                    prefix)  # this is not encouraged, as we want to fix the parameters of encoder while prefix-tuning
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
        # TODO: implement decode for graph level
        batch.x = self.decode_node_emb(batch.x)
        batch.edge_attr = self.decode_edge_emb(batch.edge_attr)
        # batch.graph_attr = self.decode_graph_emb(batch.graph_attr)
        for _ in range(self.num_decode_layers):
            batch = self.decode_layers[_](batch)
        if self.force_undirected:
            A = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
            A = (A + A.permute(0, 2, 1, 3)).reshape(-1, A.shape[-1])
            mask = A.any(dim=1)
            batch.edge_attr = A[mask]
            assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
        return self.decode_node(batch.x), self.decode_edge(batch.edge_attr), self.decode_graph(
            self.decode_graph_emb(batch.graph_attr)).flatten()

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


@register_network('GraphTransformerStructureEncoder')
class GraphTransformerStructureEncoder(nn.Module):
    """
        Full Graph Transformer Encoder; encode structural information by four types of reconstruction losses
    """

    def __init__(self, dim_in=0, dim_out=0,  # in_dim, posenc_in_dim, posenc_dim, prefix_dim, hid_dim, out_dim,
                 #              use_time, temb_dim, num_heads, num_layers, pool,
                 #              dropout=0.0,
                 #              attn_dropout=0.0,
                 #              layer_norm=False, batch_norm=True,
                 #              residual=True,
                 #              act='relu',
                 #              norm_e=True,
                 #              O_e=True,
                 #              ff_e=False,
                 cfg=None,  # need to explicitly enter cfg.encoder or cfg.cond
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

        self.node_dict_dim = cfg.get('node_encoder_num_types', 1)  # TODO: update this dimension of dictionary
        self.edge_dict_dim = cfg.get('edge_encoder_num_types', 1)
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
                pe_raw_norm_node = nn.BatchNorm1d(self.posenc_in_dim, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_node = nn.LayerNorm(self.posenc_in_dim)
            else:
                pe_raw_norm_node = nn.Identity()
            self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, self.posenc_dim))
            # self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, 2 * self.posenc_dim), self.act,
            #                                 nn.Linear(2 * self.posenc_dim, self.posenc_dim))
        if self.posenc_in_dim_edge > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_edge = nn.BatchNorm1d(self.posenc_in_dim_edge, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
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
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum) if cfg.norm_e else nn.Identity()

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
        self.decode_node_from_edge = nn.Linear(self.out_dim, self.node_dict_dim)  # virtual node
        self.decode_edge_from_node = nn.Linear(self.out_dim, self.edge_dict_dim)
        self.decode_nodes_from_edge = nn.Linear(self.out_dim, int(self.node_dict_dim * (self.node_dict_dim + 1) / 2))
        self.decode_pe_from_node = nn.Linear(self.out_dim, self.posenc_in_dim)
        self.decode_pe_from_edge = nn.Linear(self.out_dim, self.posenc_in_dim_edge)

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
        score_edge = self.act(score_edge)

        score_node = torch.zeros_like(batch.x)  # (num nodes in batch) x num_heads x out_dim
        scatter(batch.edge_attr, batch.edge_index[1], dim=0, out=score_node, reduce='add')

        return self.decode_node_from_edge(score_node), self.decode_edge_from_node(score_edge), self.decode_nodes_from_edge(batch.edge_attr), \
               self.decode_pe_from_node(batch.x), self.decode_pe_from_edge(batch.edge_attr)

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

