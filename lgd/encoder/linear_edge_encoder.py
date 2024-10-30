import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder
import logging


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10', 'planar', 'sbm', 'comm20']:
            self.in_dim = 1
        elif cfg.dataset.name in ['MUTAG']:
            self.in_dim = 4
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            # logging.info(batch)
            # logging.info(batch.shape)
            batch = self.encoder(batch.view(-1, self.in_dim))
        else:
            batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
