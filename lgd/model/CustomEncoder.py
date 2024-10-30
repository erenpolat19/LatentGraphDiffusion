import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
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


class MultiModel(torch.nn.Module):
    """Multiple layer types can be combined here.
    """

    def __init__(self, dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=None):
        super().__init__()
        if hasattr(cfg, 'in_dim'):
            dim_in = cfg.in_dim
            # the embedding dimension of the first layer encoder;
            # dimension of input data is automatically stored in the global cfg.share.dim_in
        self.encoder = FeatureEncoder(dim_in, cfg=cfg)
        dim_in = self.encoder.dim_in

        # if cfg.gnn.layers_pre_mp > 0:
        #     self.pre_mp = GNNPreMP(
        #         dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
        #     dim_in = cfg.gnn.dim_inner

        assert cfg.hid_dim == dim_in, \
            "The inner and hidden dims must match."
        self.hid_dim = cfg.hid_dim

        if hasattr(cfg, 'decode_dim'):
            dim_out = cfg.decode_dim
            # redefine the output dimension for customized tasks
        else:
            cfg.decode_dim = dim_out

        self.task = cfg.get('task', 'graph')
        assert self.task in ['node', 'edge', 'graph']
        self.task_type = cfg.get('task_type', 'classification')
        assert self.task_type in ['regression', 'classification']
        if cfg.get('label_raw_norm', None) == 'BatchNorm':
            label_norm = nn.BatchNorm1d(self.hid_dim)
        elif cfg.get('label_raw_norm', None) == 'LayerNorm':
            label_norm = nn.LayerNorm(self.hid_dim)
        else:
            label_norm = nn.Identity()
        assert cfg.get('num_task', 1) == 1  # currently only consider one task each time
        if self.task_type == 'regression':
            self.label_emb = nn.Sequential(nn.Linear(1, 2 * self.hid_dim), self.act,
                                           nn.Linear(2 * self.hid_dim, self.hid_dim), label_norm, self.act)
        else:
            self.label_emb = nn.Sequential(nn.Embedding(cfg.decode_dim + 1, self.hid_dim), label_norm)

        self.label_embed_type = cfg.get('label_embed_type', 'add_all')

        try:
            model_types = cfg.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.layer_type}")
        layers = []
        for _ in range(cfg.num_layers):
            layers.append(MultiLayer(
                dim_h=cfg.hid_dim,
                model_types=model_types,
                num_heads=cfg.num_heads,
                pna_degrees=cfg.get('pna_degrees', None),
                equivstable_pe=False,  # cfg.get('posenc_EquivStableLapPE.enable', False),
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                layer_norm=cfg.layer_norm,
                batch_norm=cfg.batch_norm,
                bigbird_cfg=cfg.get('bigbird', None),
                exp_edges_cfg=cfg.get('prep', None)
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.final_layer_node = register.head_dict['inductive_node'](cfg.hid_dim, cfg.out_dim, cfg.get('layers_post_mp', 2), cfg.get('final_norm', False)) if cfg.get('latent_node', True) else None
        self.final_layer_edge = register.head_dict['inductive_edge'](cfg.hid_dim, cfg.out_dim) if cfg.get('latent_edge', False) else None
        graph_attr_in_dim = cfg.out_dim if cfg.get('latent_node', True) else cfg.hid_dim
        self.final_layer_graph = register.head_dict['san_graph'](graph_attr_in_dim, cfg.out_dim) if cfg.get('latent_graph', True) else None

        if hasattr(cfg, 'head'):
            GNNHead = register.head_dict[cfg.head]
            self.post_mp = GNNHead(dim_in=cfg.out_dim, dim_out=dim_out)

    def forward(self, batch, label=None, **kwargs):
        batch_num_node = batch.get('num_node_per_graph', torch.tensor([batch.num_nodes], dtype=torch.long, device=batch.x.device))
        batch_node_idx = num2batch(batch_num_node)
        batch = self.encoder(batch)
        if label is not None:
            if not torch.is_tensor(label):
                label, masked_label_idx = label
            else:
                masked_label_idx = None
            if len(label.shape) == 1:
                label = label.unsqueeze(1)
            label = self.label_emb(label)
            if len(label.shape) == 3:
                label = label.squeeze(1)
            if masked_label_idx is not None:
                label = label.clone()
                label[masked_label_idx] = 0

            if self.task == 'node':
                batch.x[batch.get(batch.split+'_mask', None)] = batch.x[batch.get(batch.split+'_mask', None)] + label
            elif self.task == 'graph':
                label_embed_node = label[batch_node_idx]
                batch.x = batch.x + label_embed_node

                # virtual_node_idx = torch.cumsum(batch_num_node, dim=0) - 1
                # batch.x[virtual_node_idx] = batch.x[virtual_node_idx] + label

                # label_embed_edge = label[batch_edge_idx]
                # e = e + label_embed_edge
            else:
                raise NotImplementedError  # TODO: finish edge

        batch = self.layers(batch)
        if self.final_layer_node is not None:
            batch = self.final_layer_node(batch, return_batch=True)
        if self.final_layer_edge is not None:
            batch = self.final_layer_edge(batch, return_batch=True)
        if self.final_layer_graph is not None:
            batch = self.final_layer_graph(batch, return_batch=True)
        return batch

    def encode(self, batch, label=None, **kwargs):
        return self.forward(batch, label=label, **kwargs)

    def decode(self, batch, **kwargs):
        return self.post_mp(batch, return_batch=False, **kwargs)


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


register_network('MultiModel', MultiModel)
register_network('SingleModel', SingleModel)
