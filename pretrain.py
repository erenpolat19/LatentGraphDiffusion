# TODO: (1) preprocess and store the dataset;
#  (2) pretrain the encoder and decoder with classification and graph regression loss, store checkpoints;

import lgd  # noqa, register custom modules
import argparse
import datetime
import os
import logging
import torch
import time
from typing import Any, Dict, Tuple
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import GraphGymModule, create_model
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

#
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str, default="zinc12")
# parser.add_argument("--task", type=str, default="star")
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--epochs', type=int, default=100,
#                     help='number of epochs to train (default: 100)')
# parser.add_argument('--num_workers', type=int, default=0,
#                     help='number of workers (default: 0)')
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--run_id', type=int, default=0)
# parser.add_argument('--drop_ratio', type=float, default=0.0, help='drop out rate')
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--cos_lr', action='store_true', default=False)
# parser.add_argument('--weight_decay', type=float, default=0.0)
# parser.add_argument('--loss', type=str, default='L1Loss')
# parser.add_argument('--in_dim', type=int, default=16)
# parser.add_argument('--hid_dim', type=int, default=16)
# parser.add_argument('--out_dim', type=int, default=1)
# parser.add_argument('--num_layers', type=int, default=1)
# parser.add_argument('--num_layers_id', type=int, default=0)
# parser.add_argument('--num_layers_global', type=int, default=1)
# parser.add_argument('--num_layers_regression', type=int, default=1)
# parser.add_argument('--transformer', action='store_true', default=False)
# parser.add_argument('--rw_steps', type=int, default=20)
# parser.add_argument('--se_dim', type=int, default=16)
# parser.add_argument('--se_type', type=str, default='linear')
# parser.add_argument('--num_head', type=int, default=8)
# parser.add_argument('--norm_type', type=str, default='layer')
# parser.add_argument('--cat', type=str, default='add')
# parser.add_argument('--final_concat', type=str, default='none')
# parser.add_argument('--node_pool', type=str, default='mean')
# parser.add_argument('--subgraph_pool', type=str, default='add')
# parser.add_argument('--global_pool', type=str, default='max')
# parser.add_argument('--mask_value', type=float, default=1.0)
# parser.add_argument('--num_tasks', type=int, default=1)
# parser.add_argument('--no', type=int, default=0)
# parser.add_argument('--name', type=str, default='')
# parser.add_argument('--ensemble_train', action='store_true', default=False)
# parser.add_argument('--ensemble_test', action='store_true', default=False)
# parser.add_argument('--sample_times', type=int, default=1)
# parser.add_argument('--ensemble_mode', type=str, default='mean')
# parser.add_argument('--factor', type=float, default=0.9)  # 0.5
# parser.add_argument('--patience', type=int, default=5)  # 3
#
# args = parser.parse_args()


class GraphGymModule_custom(LightningModule):
    def __init__(self, dim_in, dim_out, cfg, enter_cfg=None):
        super().__init__()
        self.cfg = cfg
        if enter_cfg is None:
            enter_cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out,
                                                  cfg=enter_cfg)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        pred, true = self(batch)
        loss, pred_score = compute_loss(pred, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="val")

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp

    def lr_scheduler_step(self, *args, **kwargs):
        # Needed for PyTorch 2.0 since the base class of LR schedulers changed.
        # TODO Remove once we only want to support PyTorch Lightning >= 2.0.
        return super().lr_scheduler_step(*args, **kwargs)


def create_model_custom(to_device=True, dim_in=None, dim_out=None, cfg_name=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' == cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    if cfg_name is not None:
        enter_cfg = cfg.get(cfg_name, None)
        if enter_cfg is None:
            enter_cfg = cfg
    else:
        enter_cfg = cfg
    model = GraphGymModule_custom(dim_in, dim_out, cfg, enter_cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)
    # print(cfg)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        # TODO: debug loader and dataset
        # dataset = load_dataset_master(cfg.dataset.format, cfg.dataset.name, cfg.dataset.dataset_dir)
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        loaders = create_loader()
        loggers = create_logger()
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # model = create_model()
        model = create_model_custom(cfg_name='encoder')
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )
        optimizer = create_optimizer(model.parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")


