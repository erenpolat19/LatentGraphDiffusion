import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.graphgym.register import act_dict, register_layer
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
import warnings
from lgd.model.GraphTransformerEncoder import GTE_Attention, GraphTransformerEncoderLayer
from lgd.model.utils import pyg_softmax, num2batch, get_timestep_embedding, get_log_deg
from torch_geometric.graphgym.config import cfg
import logging
# from watchpoints import watch
# watch.config(pdb=True)


# TODO: preprocess the prompt examples to:
#  (1) node level, (num_node) x n_prompt x num_heads x out_dim;
#  (2) edge level, (num_node^2) x n_prompt x num_heads x out_dim;
class Cross_Attention(nn.Module):
    """
        Self-attention in graph transformer encoder
    """

    def __init__(self, in_dim, out_dim, condition, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 score_act=True,
                 signed_sqrt=False,
                 scaled_attn=True,
                 cond_alpha=1.0,
                 attn_product='mul',
                 attn_reweight=False,
                 edge_reweight=False,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        assert condition in ['masked_graph', 'prompt_graph', 'prefix', 'class_label', 'multi-modal', 'unconditional']
        self.condition = condition
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance
        self.score_act = score_act
        self.signed_sqrt = signed_sqrt
        self.cond_alpha = cond_alpha
        self.scaled_attn = scaled_attn
        assert attn_product in ['add', 'mul']
        self.attn_product = attn_product
        self.attn_reweight = attn_reweight
        self.edge_reweight = edge_reweight

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)  # TODO: should be different for different tasks?
        self.E1 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.E2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.H = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E1.weight)
        nn.init.xavier_normal_(self.E2.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.H.weight)

        if self.condition == 'masked_graph':
            self.G1 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.G2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            nn.init.xavier_normal_(self.G1.weight)
            nn.init.xavier_normal_(self.G2.weight)

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

    def propagate_attention(self, batch, type='node_wise'):
        if type == 'node_wise':  # node-wise
            # TODO: finish edge enhance
            src = batch.K[batch.edge_index[0]]  # (num_nodes) x num_heads x out_dim
            dest = batch.Q_h[batch.edge_index[1]]  # (num_nodes) x num_heads x out_dim
            score = torch.mul(src, dest) if self.attn_product == 'mul' else (src + dest)
            # isnan = torch.isnan(score).any()
            # if isnan:
            #     logging.info("nan, 8")
            #     score[torch.isnan(score)] = 0

            batch.E = batch.E.view(-1, self.num_heads, self.out_dim)
            score_e = score * batch.E
            if self.signed_sqrt:
                score_e = torch.sqrt(torch.relu(score_e).clamp_min(1e-8)) - torch.sqrt(torch.relu(-score_e).clamp_min(1e-8))
            score_e = score_e + batch.get('E_prompt', 0.) + batch.get('G_prompt_e', 0.)

            if self.score_act:
                score_e = self.act(score_e)
            e_t = score_e
            batch.wE = score_e.flatten(1)

            score_h = torch.einsum("ehd, dhc->ehc", score_e, self.Aw) if self.attn_reweight else score_e.sum(-1, keepdims=True)
            if self.scaled_attn:
                score_h = score_h / torch.sqrt(torch.tensor([self.out_dim], dtype=torch.float, device=batch.x.device))

            if self.clamp is not None:
                score_h = torch.clamp(score_h, min=-self.clamp, max=self.clamp)
            score_h = pyg_softmax(score_h, batch.edge_index[1], batch.num_nodes)  # (num node) x num_heads x 1

            msg = batch.V[batch.edge_index[0]] * score_h
            batch.wV = scatter_add(msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes)
            batch.wV = batch.wV + self.cond_alpha * (batch.get("V_prompt", 0.) + batch.get('G_prompt_h', 0.))

            if self.edge_enhance:
                rowV = torch.einsum("nhd, dhc -> nhc", e_t, self.VeRow) if self.edge_reweight else e_t
                rowV = scatter(rowV * score_h, batch.edge_index[1], dim=0, reduce="add")
                # rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
                # rowV = torch.einsum("nhd, dhc -> nhc", rowV, self.VeRow)
                batch.wV = batch.wV + rowV

        else:
            # TODO: here suppose the prompts are all graph-level, i.e. applies to all nodes. condition types include:
            #  prompt_graph, prefix, label, multi-modal
            n_prompt = batch.K.shape[0] if type == 'share' else batch.K.shape[1]
            num_edges = batch.E.shape[0]
            num_nodes = batch.num_nodes
            if type == 'share':
                src_h = batch.K.unsqueeze(0).repeat(num_nodes, 1, 1, 1)  # num_nodes in batch x n_prompt x num_heads x out_dim
                src_e = batch.K.unsqueeze(0).repeat(num_edges, 1, 1, 1)  # num_edges in batch x n_prompt x num_heads x out_dim
                v_h = batch.V.unsuqeeze(0).repeat(num_nodes, 1, 1, 1).reshape(-1, self.num_heads, self.out_dim)
                v_e = batch.V.unsuqeeze(0).repeat(num_edges, 1, 1, 1).reshape(-1, self.num_heads, self.out_dim)
            else:
                # TODO: preprocess the num_node_per_graph (including virtual nodes)
                batch_node = batch.batch_node_idx  # num2batch(batch.num_node_per_graph)
                batch_edge = batch.batch_edge_idx
                src_h = batch.K[batch_node]
                src_e = batch.K_e[batch_edge]
                v_h = batch.V[batch_node].reshape(-1, self.num_heads, self.out_dim)
                v_e = batch.V_e[batch_edge].reshape(-1, self.num_heads, self.out_dim)
            # TODO: K, V are supported to share for all graphs (e.g. shared prompt_graph, prefix or multi-modal for all graphs),
            #  or specific for each graph (e.g. multi-modal or label for each graph)

            dest_h = batch.Q_h.unsqueeze(1).repeat(1, n_prompt, 1, 1)     # num_nodes in batch x n_prompt x num_heads x out_dim
            score_h = torch.mul(src_h, dest_h)      # element-wise multiplication;
            # TODO: some other solutions including add, matrix multiplication or global attention; also consider symmetry
            if self.score_act:
                score_h = self.act(score_h)
            score_h = score_h.sum(-1).reshape(-1, self.num_heads, 1)   # (num node in batch x n_prompt) x num_heads x 1
            if self.scaled_attn:
                score_h = score_h / torch.sqrt(torch.tensor([self.out_dim], dtype=torch.float, device=batch.x.device))
            # TODO: check whether need another parameters, or simply sum over out_dim in every head
            if self.clamp is not None:
                score_h = torch.clamp(score_h, min=-self.clamp, max=self.clamp)

            idx = torch.arange(num_nodes, device=score_h.device).unsqueeze(1).repeat(1, n_prompt).reshape(-1)
            score_h = pyg_softmax(score_h, idx, num_nodes)
            score_h = self.dropout(score_h)

            msg = v_h * score_h  # (num node in batch x n_prompt) x num_heads x out_dim
            batch.wV = scatter_add(msg, idx, dim=0, dim_size=num_nodes)  # (num nodes in batch) x num_heads x out_dim

            # if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim)
            dest_e = batch.E.unsqueeze(1).repeat(1, n_prompt, 1, 1)  # num_edges in batch x n_prompt x num_heads x out_dim
            score_e = torch.mul(src_e, dest_e)
            if self.score_act:
                score_e = self.act(score_e)
            score_e = score_e.sum(-1).reshape(-1, self.num_heads, 1)  # (num_edges in batch x n_prompt) x num_heads x 1
            if self.scaled_attn:
                score_e = score_e / torch.sqrt(torch.tensor([self.out_dim], dtype=torch.float, device=batch.x.device))
            if self.clamp is not None:
                score_e = torch.clamp(score_e, min=-self.clamp, max=self.clamp)
            idx_e = torch.arange(num_edges, device=score_e.device).unsqueeze(1).repeat(1, n_prompt).reshape(-1)
            score_e = pyg_softmax(score_e, idx_e, num_edges)
            score_e = self.dropout(score_e)
            msg = v_e * score_e  # (num edges in batch x n_prompt) x num_heads x out_dim
            batch.wE = scatter_add(msg, idx_e, dim=0, dim_size=num_edges).flatten(1)  # (num edges in batch) x (num_heads x out_dim)

    def forward(self, batch, prompt=None):
        if prompt is None:
            prompt = batch.x
        if self.condition == 'masked_graph':  # TODO: consider prompt_g
            prompt, prompt_e, prompt_g = prompt
            batch.E = self.E1(prompt_e + batch.edge_attr)
            batch.E_prompt = self.act(self.E2(prompt_e)).view(-1, self.num_heads, self.out_dim)
            batch.V_prompt = self.act(self.H(prompt)).view(-1, self.num_heads, self.out_dim)
            batch.G_prompt_h = self.act(self.G1(prompt_g))[batch.batch_node_idx].view(-1, self.num_heads, self.out_dim)
            batch.G_prompt_e = self.act(self.G2(prompt_g))[batch.batch_edge_idx].view(-1, self.num_heads, self.out_dim)
        elif batch.get("edge_attr", None) is not None:  # remove this if afterwards
            batch.E = self.E1(batch.edge_attr)
        else:
            raise NotImplementedError  # this should be processed in the encoder part for unattributed edges

        Q_h = self.Q(batch.x)
        K = self.K(prompt)
        V = self.V(prompt)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)  # num_nodes x num_heads x out_dim
        if prompt.shape[0] == batch.num_nodes or self.condition in ['masked_graph', 'unconditional']:
            type = 'node_wise'
            batch.K = K.view(-1, self.num_heads, self.out_dim)  # num_nodes x num_heads x out_dim
            batch.V = V.view(-1, self.num_heads, self.out_dim)  # num_nodes x num_heads x out_dim
        elif len(prompt.shape) == 3:
            type = 'graph_wise'
            batch.K = K.view(K.shape[0], K.shape[1], self.num_heads, self.out_dim)  # batchsize x n_prompt x num_heads x out_dim
            batch.V = V.view(V.shape[0], V.shape[1], self.num_heads, self.out_dim)  # batchsize x n_prompt x num_heads x out_dim
            batch.K_e = self.E2(prompt).view(K.shape[0], K.shape[1], self.num_heads, self.out_dim)
            batch.V_e = self.H(prompt).view(V.shape[0], V.shape[1], self.num_heads, self.out_dim)
        elif len(prompt.shape) == 2:
            type = 'share'
            batch.K = K.view(-1, self.num_heads, self.out_dim)  # n_prompt x num_heads x out_dim
            batch.V = V.view(-1, self.num_heads, self.out_dim)  # n_prompt x num_heads x out_dim
            batch.K_e = self.E2(prompt).view(-1, self.num_heads, self.out_dim)
            batch.V_e = self.H(prompt).view(-1, self.num_heads, self.out_dim)
        else:
            raise NotImplementedError
        self.propagate_attention(batch, type)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


@register_layer("DT_layer")
class DenoisingTransformerLayer(nn.Module):
    """
        Full Transformer Layer of Graph Transformer Encoder
    """
    def __init__(self, in_dim, out_dim, temb_dim, condition, num_heads, final_dim=None,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 ff_e=True,
                 cond_alpha=1.0,
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
        self.condition = condition
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.ff_e = ff_e
        self.cond_alpha = cond_alpha

        # -------
        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.attn.bn_momentum
        self.bn_no_runner = cfg.attn.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", False)

        self.cross_attention = Cross_Attention(
            in_dim=out_dim,
            out_dim=out_dim // num_heads,
            condition=condition,
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
            cond_alpha=self.cond_alpha,
            attn_product=cfg.attn.get("attn_product", 'mul'),
            attn_reweight=cfg.attn.get("attn_reweight", False),
            edge_reweight=cfg.attn.get("edge_reweight", False)
        )

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

        # TODO: should have two self attention layers

        self.temb_proj_h = nn.Linear(temb_dim, in_dim)
        self.temb_proj_e = nn.Linear(temb_dim, in_dim)

        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        self.O_h2 = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e2 = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e2 = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.FFN_h_layer3 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer4 = nn.Linear(out_dim * 2, self.final_dim)

        if self.ff_e:
            self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
            self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)
            self.FFN_e_layer3 = nn.Linear(out_dim, out_dim * 2)
            self.FFN_e_layer4 = nn.Linear(out_dim * 2, self.final_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
            self.layer_norm3_h = nn.LayerNorm(out_dim)
            self.layer_norm3_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
            self.layer_norm4_h = nn.LayerNorm(self.final_dim)
            self.layer_norm4_e = nn.LayerNorm(self.final_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()
            self.batch_norm3_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm3_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()
            self.batch_norm4_h = nn.BatchNorm1d(self.final_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm4_e = nn.BatchNorm1d(self.final_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha3_h = nn.Parameter(torch.zeros(1,1))
            self.alpha4_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))
            self.alpha2_e = nn.Parameter(torch.zeros(1,1))
            self.alpha3_e = nn.Parameter(torch.zeros(1,1))
            self.alpha4_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch, temb=None, prompt=None):
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

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
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

            if self.residual:
                if self.rezero:
                    e = e * self.alpha2_e
                e = e_in2 + e
            if self.layer_norm:
                e = self.layer_norm2_e(e)
            if self.batch_norm:
                e = self.batch_norm2_e(e)

        batch.x = h
        batch.edge_attr = e
        h_in3 = h
        e_in3 = e

        # cross-attention out
        h_attn_out, e_attn_out = self.cross_attention(batch, prompt)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h2(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e2(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha3_h
            h = h_in3 + h  # residual connection
            if e is not None:
                if self.rezero:
                    e = e * self.alpha3_e
                e = e + e_in3

        if self.layer_norm:
            h = self.layer_norm3_h(h)
            if e is not None:
                e = self.layer_norm3_e(e)

        if self.batch_norm:
            h = self.batch_norm3_h(h)
            if e is not None:
                e = self.batch_norm3_e(e)

        # FFN for h
        h_in4 = h  # for second residual connection
        h = self.FFN_h_layer3(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer4(h)

        if self.residual and self.out_dim == self.final_dim:
            if self.rezero:
                h = h * self.alpha4_h
            h = h_in4 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm4_h(h)

        if self.batch_norm:
            h = self.batch_norm4_h(h)

        if self.ff_e:
            e_in4 = e
            e = self.FFN_e_layer3(e)
            e = self.act(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer4(e)

            if self.residual and self.out_dim == self.final_dim:
                if self.rezero:
                    e = e * self.alpha4_e
                e = e_in4 + e
            if self.layer_norm:
                e = self.layer_norm4_e(e)
            if self.batch_norm:
                e = self.batch_norm4_e(e)

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


class DenoisingTransformer(nn.Module):
    """
        Full Denoising Graph Transformer
    """
    def __init__(self, dim_in=0, dim_out=0, # in_dim, hid_dim, out_dim, condition_list,
    #              use_time, temb_dim, num_heads, num_layers, pool,
    #              dropout=0.0,
    #              attn_dropout=0.0,
    #              layer_norm=False, batch_norm=True,
    #              residual=True,
    #              act='relu',
    #              norm_e=True,
    #              O_e=True,
    #              ff_e=False,
    #              cfg=dict(),
                 **kwargs):
        super().__init__()

        self.in_dim = cfg.dt.in_dim
        self.hid_dim = cfg.dt.hid_dim
        self.out_dim = cfg.dt.out_dim
        for condition in cfg.dt.condition_list:
            assert condition in ['masked_graph', 'prompt_graph', 'prefix', 'class_label', 'multi-modal', 'unconditional']
        self.condition_list = cfg.dt.condition_list
        self.use_time = cfg.dt.use_time
        self.temb_dim = cfg.dt.temb_dim
        self.num_heads = cfg.dt.num_heads
        self.num_layers = cfg.dt.num_layers
        self.self_attn = cfg.dt.get('self_attn', True)
        self.dropout = cfg.dt.dropout
        self.attn_dropout = cfg.dt.attn_dropout
        self.residual = cfg.dt.residual
        self.cond_alpha = cfg.dt.get('cond_alpha', 1.0)
        self.layer_norm = cfg.dt.layer_norm
        self.batch_norm = cfg.dt.batch_norm
        self.act = act_dict[cfg.dt.act]() if cfg.dt.act is not None else nn.Identity()
        self.update_e = cfg.dt.get("update_e", True)
        self.force_undirected = cfg.dt.get("force_undirected", False)
        self.bn_momentum = cfg.dt.bn_momentum
        self.bn_no_runner = cfg.dt.bn_no_runner

        self.node_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.edge_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.cond_dim = cfg.dt.cond_dim
        self.cond_edge_dim = cfg.dt.get('cond_edge_dim', self.cond_dim)
        self.cond_in_mlp = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim), self.act)
        if 'masked_graph' in self.condition_list:
            self.cond_in_mlp_2 = nn.Sequential(nn.Linear(self.cond_edge_dim, 2 * self.hid_dim), self.act,
                                               nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim), self.act)
            self.cond_in_mlp_3 = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.hid_dim), self.act,
                                               nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim), self.act)
            self.cond_res_mlp = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.out_dim), self.act,
                                               nn.Linear(2 * self.out_dim, self.out_dim), nn.LayerNorm(self.out_dim), self.act)

        if self.layer_norm:
            self.layer_norm_in_h = nn.LayerNorm(self.hid_dim)
            self.layer_norm_in_e = nn.LayerNorm(self.hid_dim) if cfg.dt.norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum) if cfg.dt.norm_e else nn.Identity()

        self.temb_layer = nn.Sequential(nn.Linear(self.temb_dim, 2 * self.temb_dim), self.act,
                                        nn.Linear(2 * self.temb_dim, self.temb_dim))

        # TODO: here only specific one condition type, but actually capable of many types
        #  it should not be masked_graph or unconditional if want graph-level conditions
        self.denoising_layers = nn.ModuleList([
            DenoisingTransformerLayer(in_dim=self.hid_dim,
                                      out_dim=self.hid_dim,
                                      condition=self.condition_list[0],  # change this later
                                      temb_dim=self.temb_dim,
                                      num_heads=self.num_heads,
                                      dropout=self.dropout,
                                      attn_dropout=self.attn_dropout,
                                      layer_norm=self.layer_norm, batch_norm=self.batch_norm,
                                      residual=self.residual,
                                      act=cfg.dt.act,
                                      norm_e=cfg.dt.norm_e,
                                      O_e=cfg.dt.O_e,
                                      ff_e=cfg.dt.ff_e or cfg.dt.get('ff_e_ca', True),
                                      cond_alpha=self.cond_alpha,
                                      cfg=cfg.dt) for _ in range(self.num_layers)
        ])
        if self.self_attn:
            self.self_attn_layers = nn.ModuleList([
                GraphTransformerEncoderLayer(in_dim=self.hid_dim,
                                             out_dim=self.hid_dim,
                                             temb_dim=self.temb_dim,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout,
                                             attn_dropout=self.attn_dropout,
                                             layer_norm=self.layer_norm, batch_norm=self.batch_norm,
                                             residual=self.residual,
                                             act=cfg.dt.act,
                                             norm_e=cfg.dt.norm_e,
                                             O_e=cfg.dt.O_e,
                                             ff_e=cfg.dt.ff_e or cfg.dt.get('ff_e_sa', False),
                                             cfg=cfg.dt) for _ in range(self.num_layers)
            ])

        # self.final_layer_node = nn.Linear(self.hid_dim, self.out_dim)
        # self.final_layer_edge = nn.Linear(self.hid_dim, self.out_dim)
        # self.final_layer = DenoisingTransformerLayer(in_dim=self.hid_dim,
        #                                              out_dim=self.hid_dim,
        #                                              final_dim=self.out_dim,
        #                                              condition=self.condition_list[0],
        #                                              temb_dim=self.temb_dim,
        #                                              num_heads=self.num_heads,
        #                                              dropout=self.dropout,
        #                                              attn_dropout=self.attn_dropout,
        #                                              layer_norm=self.layer_norm, batch_norm=self.batch_norm,
        #                                              residual=self.residual,
        #                                              act=cfg.dt.act,
        #                                              norm_e=cfg.dt.norm_e,
        #                                              O_e=cfg.dt.O_e,
        #                                              ff_e=cfg.dt.ff_e,
        #                                              cond_alpha=self.cond_alpha,
        #                                              cfg=cfg.dt)

        self.final_layer_node = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_layer_edge = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_norm = cfg.dt.get('final_norm', False)
        if self.final_norm:
            self.final_norm_node_1 = nn.LayerNorm(self.out_dim)
            self.final_norm_edge_1 = nn.LayerNorm(self.out_dim)

        assert cfg.dt.pool in ['max', 'add', 'mean', 'none']
        self.pool = cfg.dt.pool
        self.pool_vn = cfg.dt.get('pool_vn', False)
        self.pool_edge = cfg.dt.get('pool_edge', False)
        self.post_pool = cfg.dt.get('post_pool', False)
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

    def forward(self, batch, t=None, prompt=None, **kwargs):
        # batch.x_0 = batch.x
        # batch.edge_attr_0 = batch.edge_attr
        # num_nodes, num_edges = batch.num_nodes, batch.edge_index.shape[1]

        batch_num_node = batch.num_node_per_graph
        batch_node_idx = num2batch(batch_num_node)
        assert torch.equal(batch_node_idx, batch.batch)
        batch_edge_idx = num2batch(batch_num_node ** 2)
        batch.batch_node_idx, batch.batch_edge_idx = batch_node_idx, batch_edge_idx

        h = self.node_in_mlp(batch.x)
        e = self.edge_in_mlp(batch.edge_attr)
        if 'masked_graph' in self.condition_list:  # TODO: prompt_graph, how to process? the hidden_dim are different too
            prompt_h0, prompt_e0, prompt_g0 = prompt
            prompt_h = self.cond_in_mlp(prompt_h0)
            prompt_e = self.cond_in_mlp_2(prompt_e0)
            prompt_g = self.cond_in_mlp_3(prompt_g0)
            prompt = (prompt_h, prompt_e, prompt_g)
            batch.prompt_h0, batch.prompt_e0, batch.prompt_g0 = prompt_h0, prompt_e0, prompt_g0
        elif prompt is None or 'unconditional' in self.condition_list:
            prompt = None
        else:
            prompt = self.cond_in_mlp(prompt)
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
            # logging.info(_)
            batch = self.denoising_layers[_](batch, temb, prompt)
            # isnan = torch.isnan(batch.x).any() or torch.isnan(batch.edge_attr).any()
            # if isnan:
            #     logging.info('ca')
            if self.self_attn:
                batch = self.self_attn_layers[_](batch, temb)
                # isnan = torch.isnan(batch.x).any() or torch.isnan(batch.edge_attr).any()
                # if isnan:
                #     logging.info('sa')

        batch.x = self.final_layer_node(batch.x)
        # logging.info(torch.isnan(batch.x).any())
        if self.force_undirected:
            A = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
            A = (A + A.permute(0, 2, 1, 3)).reshape(-1, A.shape[-1])
            mask = A.any(dim=1)
            batch.edge_attr = A[mask]
            assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
        batch.edge_attr = self.final_layer_edge(batch.edge_attr)
        # logging.info(torch.isnan(batch.edge_attr).any())
        if self.final_norm:
            batch.x = self.final_norm_node_1(batch.x)
            batch.edge_attr = self.final_norm_edge_1(batch.edge_attr)

        virtual_node_idx = torch.cumsum(batch.num_node_per_graph, dim=0) - 1
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
                # batch.x[virtual_node_idx] = v_g
        else:
            v_g = batch.x[virtual_node_idx]
        if 'masked_graph' in self.condition_list:
            prompt_g0 = batch.get('prompt_g0', None)
            if prompt_g0 is not None:
                v_g = v_g + self.cond_res_mlp(prompt_g0)
        batch.graph_attr = v_g
        return batch



# emb = get_timestep_embedding(torch.tensor([1, 2]), 4)
# print(emb)


