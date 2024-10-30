from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5

    cfg.name_tag = ""

    cfg.train.ckpt_best = False
    cfg.train.start_eval_epoch = -1

    cfg.train.pretrain = CN()
    cfg.train.pretrain.mask_node_prob = 0.0
    cfg.train.pretrain.mask_edge_prob = 0.0
    cfg.train.pretrain.recon = 'masked'  # assert in 'masked' and 'all'
    cfg.train.pretrain.original_task = False
    cfg.train.pretrain.input_target = False
    cfg.train.pretrain.edge_factor = 1.0
    cfg.train.pretrain.graph_factor = 1.0
    cfg.train.pretrain.atom_bond_only = True

    cfg.diffusion = CN()
    cfg.diffusion.conditioning_key = 'crossattn'
    cfg.diffusion.edge_factor = 1.0
    cfg.diffusion.graph_factor = 0.0


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False
