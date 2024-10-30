import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_max, scatter_add
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import act_dict
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from lgd.layer.multi_model_layer import MultiLayer, SingleLayer
from lgd.encoder.ER_edge_encoder import EREdgeEncoder
from lgd.encoder.exp_edge_fixer import ExpanderEdgeFixer
from .utils import *


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in=None, cfg=None):
        super(FeatureEncoder, self).__init__()
        if dim_in is None:
            dim_in = cfg.in_dim
        self.dim_in = dim_in
        if cfg.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.node_encoder_name]
            self.node_encoder = NodeEncoder(self.dim_in)
            if cfg.node_encoder_bn:
                self.node_encoder_bn = nn.BatchNorm1d(self.dim_in)

                # self.node_encoder_bn = BatchNorm1dNode(
                #     new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                #                      has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.in_dim
        if cfg.edge_encoder:
            if not hasattr(cfg, 'dim_edge') or cfg.dim_edge is None:
                cfg.dim_edge = cfg.in_dim

            # if cfg.edge_encoder_name == 'ER':
            #     self.edge_encoder = EREdgeEncoder(cfg.dim_edge)
            # elif cfg.edge_encoder_name.endswith('+ER'):
            #     EdgeEncoder = register.edge_encoder_dict[
            #         cfg.edge_encoder_name[:-3]]
            #     self.edge_encoder = EdgeEncoder(cfg.dim_edge - cfg.posenc_ERE.dim_pe)
            #     self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            # else:
            EdgeEncoder = register.edge_encoder_dict[cfg.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dim_edge)

            if cfg.edge_encoder_bn:
                self.edge_encoder_bn = nn.BatchNorm1d(cfg.dim_edge)
                # self.edge_encoder_bn = BatchNorm1dNode(
                #     new_layer_config(cfg.gt.dim_edge, -1, -1, has_act=False,
                #                     has_bias=False, cfg=cfg))

        if 'Exphormer' in cfg.layer_type:
            self.exp_edge_fixer = ExpanderEdgeFixer(add_edge_index=cfg.prep.add_edge_index,
                                                    num_virt_node=cfg.prep.num_virt_node)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class NodeDecoder(torch.nn.Module):
    """
    Decode node features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, in_dim, decode_dim):
        super(NodeDecoder, self).__init__()


class Cross_Attention_Add(nn.Module):
    """
        simplified cross-attention
    """

    def __init__(self, in_dim, out_dim, condition, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=False,
                 update_e=False,
                 score_act=True,
                 attn_product='mul',
                 batch_norm=True,
                 layer_norm=False,
                 cfg=None,
                 **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert condition in ['masked_graph', 'prompt_graph', 'prefix', 'class_label', 'multi-modal', 'unconditional']
        self.condition = condition
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance
        self.update_e = update_e
        self.score_act = score_act
        assert attn_product in ['add', 'mul']
        self.attn_product = attn_product
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        self.Q = nn.Linear(in_dim, out_dim, bias=True)
        self.K = nn.Linear(in_dim, out_dim, bias=use_bias)  # TODO: should be different for different tasks?
        self.E1 = nn.Linear(in_dim, out_dim, bias=True)
        self.E2 = nn.Linear(out_dim, out_dim, bias=True)
        self.V = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.H = nn.Linear(out_dim, out_dim, bias=True)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E1.weight)
        nn.init.xavier_normal_(self.E2.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.H.weight)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        if self.batch_norm:
            self.batch_norm_h = nn.BatchNorm1d(out_dim)
        if self.layer_norm:
            self.layer_norm_h = nn.LayerNorm(out_dim)

    def forward(self, batch, prompt=None):
        if prompt is None:
            prompt = batch.x
        if self.condition == 'masked_graph':  # TODO: consider prompt_g
            prompt, prompt_e, prompt_g = prompt
        else:
            raise NotImplementedError  # this should be processed in the encoder part for unattributed edges

        Q = self.Q(batch.x)
        K = self.K(prompt)
        V = self.V(prompt)

        if self.attn_product == 'add':
            h = Q + K
        else:
            h = Q * K
        if self.score_act:
            h = self.act(h)
        src = h[batch.edge_index[0]]
        if prompt_e is not None and self.edge_enhance:
            e = self.act(self.E1(batch.edge_attr + prompt_e))
            e = self.act(self.E2(e + prompt_e))
            msg = self.act(src + e)
            if self.update_e:
                batch.edge_attr = e
        else:
            msg = src
        aggr = torch.zeros_like(h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=aggr, reduce='add')

        h_out = self.act(self.H(h + V + self.dropout(aggr)))
        # residual connection requires in_dim == out_dim
        if self.in_dim == self.out_dim:
            h_out = h_out + batch.x
        h_out = self.act(self.FFN_h_layer1(h_out))
        h_out = self.FFN_h_layer2(self.dropout(h_out))
        batch.x = h_out

        return batch


@register_network('CustomDenoisingNetwork')
class CustomDenoisingNetwork(torch.nn.Module):
    """Multiple layer types can be combined here.
    """

    def __init__(self, dim_in=0, dim_out=0):
        super().__init__()

        self.in_dim = cfg.dt.in_dim
        self.hid_dim = cfg.dt.hid_dim
        self.out_dim = cfg.dt.out_dim
        for condition in cfg.dt.condition_list:
            assert condition in ['masked_graph', 'prompt_graph', 'prefix', 'class_label', 'multi-modal',
                                 'unconditional']
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
        self.update_e = cfg.dt.get("update_e", False)
        self.bn_momentum = cfg.dt.bn_momentum
        self.bn_no_runner = cfg.dt.bn_no_runner

        self.node_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.edge_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.cond_dim = cfg.dt.cond_dim
        self.cond_in_mlp = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim),
                                         self.act)
        if 'masked_graph' in self.condition_list:
            self.cond_in_mlp_2 = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.hid_dim), self.act,
                                               nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim),
                                               self.act)
            self.cond_in_mlp_3 = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.hid_dim), self.act,
                                               nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim),
                                               self.act)
            self.cond_res_mlp = nn.Sequential(nn.Linear(self.cond_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim), nn.LayerNorm(self.out_dim),
                                              self.act)

        if self.layer_norm:
            self.layer_norm_in_h = nn.LayerNorm(self.hid_dim)
            self.layer_norm_in_e = nn.LayerNorm(self.hid_dim) if cfg.dt.norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm_in_h = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum)
            self.batch_norm_in_e = nn.BatchNorm1d(self.hid_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                  momentum=self.bn_momentum) if cfg.dt.norm_e else nn.Identity()

        self.temb_layer = nn.Sequential(nn.Linear(self.temb_dim, 2 * self.hid_dim), self.act,
                                        nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim), self.act)
        t_layer = []
        for _ in range(self.num_layers):
            t_layer.append(nn.Sequential(nn.Linear(self.hid_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim), nn.LayerNorm(self.hid_dim), self.act))
        self.t_layers = nn.ModuleList(t_layer)
        try:
            model_types = cfg.dt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.dt.layer_type}")
        layers = []
        for _ in range(self.num_layers):
            layers.append(MultiLayer(
                dim_h=cfg.dt.hid_dim,
                model_types=model_types,
                num_heads=cfg.dt.num_heads,
                pna_degrees=cfg.dt.get('pna_degrees', None),
                equivstable_pe=False,  # cfg.get('posenc_EquivStableLapPE.enable', False),
                dropout=cfg.dt.dropout,
                attn_dropout=cfg.dt.attn_dropout,
                layer_norm=cfg.dt.layer_norm,
                batch_norm=cfg.dt.batch_norm,
                bigbird_cfg=cfg.dt.get('bigbird', None),
                exp_edges_cfg=cfg.dt.get('prep', None)
            ))
        self.layers = torch.nn.ModuleList(layers)
        cross_layer = []
        for _ in range(self.num_layers):
            cross_layer.append(Cross_Attention_Add(
                in_dim=cfg.dt.hid_dim,
                out_dim=cfg.dt.hid_dim,
                condition=self.condition_list[0],
                num_heads=cfg.dt.num_heads,
                use_bias=True,
                dropout=cfg.dt.dropout,
                act=cfg.dt.act,
                edge_enhance=cfg.dt.attn.get('edge_enhance', False),
                update_e=self.update_e,
                attn_product='mul',
                batch_norm=self.batch_norm,
                layer_norm=self.layer_norm))
        self.cross_layers = torch.nn.ModuleList(cross_layer)

        self.final_layer_node = register.head_dict['inductive_node'](cfg.dt.hid_dim, cfg.dt.out_dim, cfg.dt.get('layers_post_mp', 2), cfg.dt.get('final_norm', False)) if cfg.dt.get('latent_node', True) else None
        self.final_layer_edge = register.head_dict['inductive_edge'](cfg.dt.hid_dim, cfg.dt.out_dim) if cfg.dt.get('latent_edge', False) else None
        graph_attr_in_dim = cfg.dt.out_dim if cfg.dt.get('latent_node', True) else cfg.dt.hid_dim
        self.final_layer_graph = register.head_dict['san_graph'](graph_attr_in_dim, cfg.dt.out_dim) if cfg.dt.get('latent_graph', False) else None

        # if hasattr(cfg, 'head'):
        #     GNNHead = register.head_dict[cfg.head]
        #     self.post_mp = GNNHead(dim_in=cfg.out_dim, dim_out=dim_out)

    def forward(self, batch, t=None, prompt=None, **kwargs):
        batch_num_node = batch.get('num_node_per_graph', torch.tensor([batch.num_nodes], dtype=torch.long, device=batch.x.device))
        batch_node_idx = num2batch(batch_num_node)
        h = self.node_in_mlp(batch.x)
        e = self.edge_in_mlp(batch.edge_attr)
        if 'masked_graph' in self.condition_list:  # TODO: prompt_graph, how to process? the hidden_dim are different too
            prompt_h0, prompt_e0, prompt_g0 = prompt
            prompt_h = self.cond_in_mlp(prompt_h0)
            prompt_e = self.cond_in_mlp_2(prompt_e0) if prompt_e0 is not None else None
            prompt_g = self.cond_in_mlp_3(prompt_g0) if prompt_g0 is not None else None
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
            batch.temb = temb

        for _ in range(self.num_layers):
            if batch.get('temb', None) is not None:
                temb = self.t_layers[_](batch.temb)
                temb_h = temb[batch_node_idx]
                batch.x = batch.x + temb_h
            batch = self.cross_layers[_](batch, prompt=prompt)
            batch = self.layers[_](batch)
        if self.final_layer_node is not None:
            batch = self.final_layer_node(batch, return_batch=True)
            batch.x = batch.x + self.cond_res_mlp(batch.prompt_h0)
        if self.final_layer_edge is not None:
            batch = self.final_layer_edge(batch, return_batch=True)
        if self.final_layer_graph is not None:
            batch = self.final_layer_graph(batch, return_batch=True)
            if batch.get('prompt_g0', None) is not None:
                batch.graph_attr = batch.graph_attr + self.cond_res_mlp(batch.prompt_g0)
        return batch

    def encode(self, batch, t=None, prompt=None, **kwargs):
        return self.forward(batch, t=t, prompt=prompt, **kwargs)


class SingleModel(torch.nn.Module):
    """A single layer type can be used without FFN between the layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(SingleLayer(
                dim_h=cfg.gt.dim_hidden,
                model_type=cfg.gt.layer_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                exp_edges_cfg=cfg.prep
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# register_network('MultiModel', MultiModel)
# register_network('SingleModel', SingleModel)
