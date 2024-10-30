import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from lgd.loss.subtoken_prediction_loss import subtoken_cross_entropy
from lgd.asset.utils import cfg_to_dict, flatten_dict, make_wandb_name, mlflow_log_cfgdict
from copy import deepcopy
import warnings
from utils import random_mask


def pretrain_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
            node_label = batch.y.clone().detach()[batch.get('train_mask', None)].flatten()
            edge_label = batch.get('edge_label', None)
            if cfg.dataset.task == 'node':
                label = node_label
            elif cfg.dataset.task == 'edge':
                label = edge_label
            else:
                raise NotImplementedError
        # TODO: elif cfg.dataset.format in ['PyG-Amazon', 'PyG-Coauthor', 'PyG-TUDataset']:
        else:
            label = batch.y.clone().detach()

        # node_label, edge_label, graph_label = batch.x.clone().detach(), batch.edge_attr.clone().detach(), batch.y
        # if len(graph_label.shape) == 1:
        #     graph_label = graph_label.unsqueeze(1)  # for qm9
        # if cfg.dataset.get("align", False):
        #     node_label = batch.x_simplified
        # if not cfg.train.pretrain.atom_bond_only:
        #     node_label = node_label[:, 0].flatten()
        #     edge_label = edge_label[:, 0].flatten()
        # else:
        #     node_label, edge_label = node_label.flatten(), edge_label.flatten()
        # batch, masked_node_idx, masked_edge_idx = random_mask(batch, cfg.train.pretrain.mask_node_prob, cfg.train.pretrain.mask_edge_prob)
        # the embed of labels and prefix are done in fine-tuning of the encoder, not pretraining
        if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
            if cfg.dataset.task == 'node':
                input_label = batch.y.clone().detach()[batch.get('train_mask', None)] if cfg.train.pretrain.input_target else None
            elif cfg.dataset.task == 'edge':
                input_label = batch.get('edge_label', None) if cfg.train.pretrain.input_target else None  # TODO: reconsider the format of input labels
            else:
                raise NotImplementedError
        # TODO: elif cfg.dataset.format in ['PyG-Amazon', 'PyG-Coauthor', 'PyG-TUDataset']:
        else:
            input_label = batch.y.clone().detach() if cfg.train.pretrain.input_target else None
        if input_label is not None:
            masked_label_idx = torch.rand(input_label.shape[0]) < cfg.train.pretrain.get("mask_label_prob", 0.)
            input_label = (input_label, masked_label_idx)
        pred = model(batch, label=input_label)
        if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
            pred_decode = model.model.decode(pred, task=cfg.dataset.task, split=batch.split)
        else:
            assert cfg.dataset.name in ['ogbn-arxiv', 'ogbn-proteins']
            pred_decode = model.model.decode(pred)
        loss, _ = compute_loss(pred_decode, label)

        if cfg.train.pretrain.get('reg', False):
            criterion_reg = nn.MSELoss()
            loss_reg = criterion_reg(pred.x, torch.zeros(pred.x.shape, device=pred.x.device)).mean()  #+ \
                       #criterion_reg(pred.edge_attr, torch.zeros(pred.edge_attr.shape, device=pred.edge_attr.device)).mean()
            loss = loss + loss_reg.mean() * cfg.train.pretrain.get('reg_factor', 1e-5)

        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        _true = label.detach().to('cpu', non_blocking=True)
        _pred = pred_decode.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', repeat=1, ensemble_mode='none', logger_1=None):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            if ensemble_mode == 'none':
                # pred, true = model(batch)
                # node_label, edge_label, graph_label = batch.x.clone().detach(), batch.edge_attr.clone().detach(), batch.y
                if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
                    node_label = batch.y.clone().detach()[batch.get(split+'_mask', None)].flatten()
                    edge_label = batch.get('edge_label', None)
                    if cfg.dataset.task == 'node':
                        label = node_label
                    elif cfg.dataset.task == 'edge':
                        label = edge_label
                    else:
                        raise NotImplementedError
                # TODO: elif cfg.dataset.format in ['PyG-Amazon', 'PyG-Coauthor', 'PyG-TUDataset']:
                else:
                    label = batch.y.clone().detach()
                if cfg.train.pretrain.input_target:
                    batch_1 = copy.deepcopy(batch)
                    # input_label = batch_1.y.clone().detach()
                    if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
                        if cfg.dataset.task == 'node':
                            input_label = batch_1.y.clone().detach()[batch_1.get(split+'_mask', None)]
                        elif cfg.dataset.task == 'edge':
                            input_label = batch_1.get('edge_label', None)  # TODO: reconsider the format of input labels
                        else:
                            raise NotImplementedError
                    # TODO: elif cfg.dataset.format in ['PyG-Amazon', 'PyG-Coauthor', 'PyG-TUDataset']:
                    else:
                        input_label = batch_1.y.clone().detach()
                    pred_ = model(batch_1, label=input_label)
                    if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
                        pred_decode_ = model.model.decode(pred_, task=cfg.dataset.task, split=batch.split)
                    else:
                        assert cfg.dataset.name in ['ogbn-arxiv', 'ogbn-proteins']
                        pred_decode_ = model.model.decode(pred_)
                    loss_labeled, _ = compute_loss(pred_decode_, label)
                    if logger_1 is not None:
                        extra_stats = {}
                        logger_1.update_stats(true=label.detach().to('cpu', non_blocking=True),
                                              pred=pred_decode_.detach().to('cpu', non_blocking=True),
                                              loss=loss_labeled.detach().cpu().item(),
                                              lr=0, time_used=time.time() - time_start,
                                              params=cfg.params,
                                              dataset_name=cfg.dataset.name,
                                              **extra_stats)
                pred = model(batch, label=None)
                if cfg.dataset.format in ['PyG-Planetoid', 'PyG-Amazon', 'PyG-Coauthor'] or cfg.dataset.transductive:
                    pred_decode = model.model.decode(pred, task=cfg.dataset.task, split=batch.split)
                else:
                    assert cfg.dataset.name in ['ogbn-arxiv', 'ogbn-proteins']
                    pred_decode = model.model.decode(pred)
            else:
                raise NotImplementedError
                # batch_pred = []
                # for i in range(repeat):
                #     bc = deepcopy(batch)
                #     pred, true = model(bc)
                #     batch_pred.append(pred)
                #     del bc
                # batch_pred = torch.cat(batch_pred).reshape(repeat, -1)
                # if ensemble_mode == 'mean':
                #     pred = torch.mean(batch_pred, dim=0)
                # else:
                #     pred = torch.median(batch_pred, dim=0)[0]
            # pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            true = label  # TODO: check this
            loss, pred_score = compute_loss(pred_decode, true)
            loss_labeled = loss_labeled if cfg.train.pretrain.input_target else torch.tensor(0.)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            loss_labeled=loss_labeled.detach().cpu().item(),
                            # loss_recon=loss_recon.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


@register_train('pretrain_encoder_inductive')
def custom_pretrain_encoder(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        pretrain_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                logger_1 = copy.deepcopy(loggers[i]) if cfg.train.pretrain.input_target else None
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1], repeat=cfg.train.ensemble_repeat, ensemble_mode=cfg.train.ensemble_mode, logger_1=logger_1)
                if logger_1 is not None:
                    loggers[i]._custom_stats['metric_labeled'] = logger_1.metric_stat(cfg.metric_best) * logger_1._size_current
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and is_ckpt_epoch(cur_epoch): # and not cfg.train.ckpt_best:
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)
