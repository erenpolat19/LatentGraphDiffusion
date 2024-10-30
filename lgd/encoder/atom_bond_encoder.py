import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

@register_node_encoder('Atom_pad')
class AtomEncoder_pad(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
        num_classes: None
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim + 10, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        encoded_features = 0
        if isinstance(x, torch.Tensor):
            for i in range(x.shape[1]):
                encoded_features += self.atom_embedding_list[i](x[:, i])
            return encoded_features
        else:
            for i in range(x.x.shape[1]):
                encoded_features += self.atom_embedding_list[i](x.x[:, i])
            x.x = encoded_features
            return x



@register_edge_encoder('Bond_pad')
class BondEncoder_pad(torch.nn.Module):
    """
    The bond Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """
    def __init__(self, emb_dim):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim + 10, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        if isinstance(edge_attr, torch.Tensor):
            for i in range(edge_attr.shape[1]):
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

            return bond_embedding
        else:
            for i in range(edge_attr.edge_attr.shape[1]):
                bond_embedding += self.bond_embedding_list[i](edge_attr.edge_attr[:, i])
            edge_attr.edge_attr = bond_embedding
            return edge_attr