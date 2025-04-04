import torch
import argparse
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, set_cfg, load_cfg)
from lgd.ddpm.LGD import DDPM, LatentDiffusion
from lgd.ddpm.LGD_Inductive import LatentDiffusionInductive

import lgd  # noqa, register custom modules
import argparse
import datetime
import os
import logging
import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict, network_dict, register_network, act_dict
from torch_geometric import seed_everything
from lgd.asset.logger import create_logger
from lgd.loader.master_loader import load_dataset_master
from lgd.optimizer.extra_optimizers import ExtendedSchedulerConfig
from lgd.agg_runs import agg_runs
from lgd.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from lgd.ddpm.LGD import DDPM, LatentDiffusion
from lgd.ddpm.LGD_Inductive import LatentDiffusionInductive
from lgd.train.pretrain_encoder import *
from lgd.encoder.atom_bond_encoder import *
from lgd.model.GraphTransformerEncoder import GraphTransformerEncoder # HERE 
from lgd.train.train_diffusion import *
from utils import print_gpu_usage
import sys
import torch.nn.functional as F
from pretrain import *
from lgd.model.GraphTransformerEncoder import *

def load_model(checkpoint_path, device="cuda"):
    """Load the trained model from a checkpoint using the global `cfg` object."""
    
    # Set device
    cfg.accelerator = device
    cfg.devices = 0  # Use the first GPU
    
    # Initialize the model
    model_type = cfg.model.get('type', 'LatentDiffusion')
    model = eval(model_type)(
        timesteps=cfg.diffusion.get('timesteps', 1000),
        conditioning_key=cfg.diffusion.conditioning_key,
        hid_dim=cfg.diffusion.hid_dim,
        parameterization=cfg.diffusion.get("parameterization", "x0"),
        cond_stage_key=cfg.diffusion.cond_stage_key,
        first_stage_config=cfg.diffusion.first_stage_config,
        cond_stage_config=cfg.diffusion.cond_stage_config,
        edge_factor=cfg.diffusion.get("edge_factor", 1.0),
        graph_factor=cfg.diffusion.get("graph_factor", 1.0),
        train_mode=cfg.diffusion.get("train_mode", 'sample')
    ).to(device)
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model

def generate_graphs(model, batch, cond, num_samples=10, device="cuda"):
    """Generate graphs using the trained model."""
    batch = batch.to(device)
    cond = cond.to(device) if cond is not None else None

    # print(batch)
    # print("Initial batch.edge_attr shape:", batch.edge_attr.shape)

    num_node = batch.num_node_per_graph
    batch_idx = torch.cat([batch.batch, num2batch(num_node ** 2)], dim=0)

    generated_graphs = model.sample(batch, cond, batch_size=num_samples, batch_idx=batch_idx) #, shape=(batch.num_nodes + batch.edge_index.shape[1], 4)) # HERE -- TODO: FIX! Hardcoded feature_dim = 4
    return generated_graphs

def visualize_graphs(graphs, num_visualizations=5, title=None):
    """Visualize generated graphs using NetworkX and Matplotlib."""
    num_visualizations = min(num_visualizations, len(graphs))
    
    fig, axes = plt.subplots(1, num_visualizations, figsize=(5 * num_visualizations, 5))
    if num_visualizations == 1:
        axes = [axes]  # Ensure iterable
    
    for i, data in enumerate(graphs[:num_visualizations]):
        G = to_networkx(data, to_undirected=True)
        nx.draw(G, with_labels=False, node_size=30, ax=axes[i])
        axes[i].set_title(f"Generated Graph {i+1}")
    
    plt.savefig(f"{title}.png")

if __name__ == "__main__":
    # Load cmd line args
    args = parse_args()
    
    # Load config file
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)
    
    # dump_cfg(cfg)

    loaders = create_loader()
    train_loader, val_loader, test_loader = loaders
    print(cfg)

    # dataset = load_dataset_master(cfg.dataset.format, cfg.dataset.name, cfg.dataset.dataset_dir) # HERE - uncommented this line

    model_enc = eval(cfg.model.get('type', 'LatentDiffusion'))\
            (timesteps=cfg.diffusion.get('timesteps', 1000), conditioning_key=cfg.diffusion.conditioning_key,
             hid_dim=cfg.diffusion.hid_dim, parameterization=cfg.diffusion.get("parameterization", "x0"),
             cond_stage_key=cfg.diffusion.cond_stage_key, first_stage_config=cfg.diffusion.first_stage_config,
             cond_stage_config=cfg.diffusion.cond_stage_config, edge_factor=cfg.diffusion.get("edge_factor", 1.0),
             graph_factor=cfg.diffusion.get("graph_factor", 1.0),
             train_mode=cfg.diffusion.get("train_mode", 'sample')).to(torch.device(cfg.accelerator))
    encoder_model = model_enc.first_stage_model
    print(encoder_model.training) 

    # Load the trained model
    model_diff = load_model(cfg.pretrained.dir)

    num_samples = 10

    for i, batch in enumerate(test_loader):
    # Generate graphs
        generated_graphs = generate_graphs(model_diff, batch, cond=None, num_samples=10)

    # Visualize the first few generated graphs
    visualize_graphs(generated_graphs, num_visualizations=5, title='gen_graph_vis_CF_450')
    visualize_graphs(test_loader, num_visualizations=5, title='orig_graph_vis_CF_450')
