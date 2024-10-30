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
import math
import warnings
from inspect import isfunction
import importlib


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    # num_nodes = maybe_num_nodes(index, num_nodes)
    # out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    # TODO: check whether +1 in softmax is necessary to output near-zero attention;
    #  also should not minus maximum in this case
    out = src
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1.)

    return out


def num2batch(num_node: Tensor):
    offset = cumsum_pad0(num_node)
    # print(offset.shape, num_subg.shape, offset[-1] + num_subg[-1])
    batch_idx = torch.zeros((offset[-1] + num_node[-1]),
                        device=offset.device,
                        dtype=offset.dtype)
    batch_idx[offset] = 1
    batch_idx[0] = 0
    batch_idx = batch_idx.cumsum_(dim=0)
    return batch_idx


def cumsum_pad0(num: Tensor):
    ret = torch.empty_like(num)
    ret[0] = 0
    ret[1:] = torch.cumsum(num[:-1], dim=0)
    return ret


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        # TODO: modify edge_index to real edges; now it has been complete graph
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_into_sparse_tensor(a, t, num_node):
    b, *_ = t.shape
    out = a.gather(-1, t)
    if num_node.shape[0] == b:
        idx = torch.cat([num2batch(num_node), num2batch(num_node ** 2)], dim=0)
    else:
        idx = num_node
    return out[idx].unsqueeze(1)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Prefix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.prefix = nn.Parameter(torch.zeros([dim]), requires_grad=True)

    def forward(self):
        return self.prefix


def symmetrize_tensor(tensor, offset=1, scale=torch.sqrt(torch.tensor(2.0))):
    B, N, _, d = tensor.shape

    # for Gaussian noise, we need offset=1
    i, j = torch.triu_indices(N, N, offset=offset)

    # symmetrize
    tensor[:, i, j, :] += tensor[:, j, i, :]
    # scale the noises to remain as standard Gaussian
    tensor[:, i, j, :] /= scale
    tensor[:, j, i, :] = tensor[:, i, j, :]

    return tensor


def symmetrize(edge_index, batch, tensor, offset=1, scale=torch.sqrt(torch.tensor(2.0))):
    A = to_dense_adj(edge_index, batch, tensor)
    A = symmetrize_tensor(A, offset, scale)
    A = A.reshape(-1, A.shape[-1])
    mask = A.any(dim=1)
    symmetrized = A[mask]
    assert symmetrized.shape[0] == edge_index.shape[1]
    return symmetrized