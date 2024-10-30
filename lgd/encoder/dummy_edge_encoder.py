import torch
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('DummyEdge')
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            dummy_attr = batch.new_zeros(batch.shape[0])
            return self.encoder(dummy_attr)
        else:
            dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
            batch.edge_attr = self.encoder(dummy_attr)
            return batch
