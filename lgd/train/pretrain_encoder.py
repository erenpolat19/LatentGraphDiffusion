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


def pretrain_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, visualize=False):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()

    tuple_label_dict = create_label_mapping(model.model.node_dict_dim, device=torch.device(cfg.accelerator))

    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        node_label, edge_label, graph_label = batch.x.clone().detach(), batch.edge_attr.clone().detach(), batch.y
        # if len(graph_label.shape) == 1:
        #     graph_label = graph_label.unsqueeze(1)  # for qm9
        if cfg.dataset.get("align", False):
            node_label = batch.x_simplified
        if not cfg.train.pretrain.atom_bond_only:
            node_label = node_label[:, 0].flatten()
            edge_label = edge_label[:, 0].flatten()
        else:
            node_label, edge_label = node_label.flatten(), edge_label.flatten()
        batch, masked_node_idx, masked_edge_idx = random_mask(batch, cfg.train.pretrain.mask_node_prob,
                                                              cfg.train.pretrain.mask_edge_prob)
        # the embed of labels and prefix are done in fine-tuning of the encoder, not pretraining
        masked_label_idx = torch.rand(batch.num_graphs) < cfg.train.pretrain.get("mask_label_prob", 0.)
        input_label = batch.y.clone().detach() if cfg.train.pretrain.input_target else None
        if input_label is not None:
            if cfg.dataset.format == 'PyG-QM9':
                input_label = (input_label - batch.y_mean) / batch.y_std
            input_label = (input_label, masked_label_idx)
        pred = model(batch, label=input_label)
        if visualize and iter == 0:
            logging.info(node_label[:20])
            logging.info(pred.x[:20])
            logging.info(edge_label[:20])
            logging.info(pred.edge_attr[:20])
        if cfg.train.pretrain.get("freeze_encoder", False):
            pred.x = pred.x.detach()
            pred.edge_attr = pred.edge_attr.detach()
            pred.graph_attr = pred.graph_attr.detach()
        node_pred, edge_pred, graph_pred = model.model.decode(pred)
        # if iter == 0:
        #     logging.info(cfg.train.pretrain.input_target)
        #     logging.info(masked_label_idx.float().sum())
        #     logging.info(masked_node_idx)
        #     logging.info(node_pred[masked_node_idx].shape)
        #     logging.info(node_label[masked_node_idx].shape)
        if cfg.dataset.format == 'PyG-QM9':
            graph_pred = graph_pred * batch.get('y_std', 1.) + batch.get('y_mean', 0.)
        criterion_node = nn.CrossEntropyLoss()
        criterion_edge = nn.CrossEntropyLoss()
        # TODO: the original task loss should be modified; L1Loss works for zinc, PCQM4Mv2 and QM9
        assert cfg.train.pretrain.recon in ['all', 'masked', 'none']
        loss_node = criterion_node(node_pred, node_label) if cfg.train.pretrain.recon == 'all' \
            else criterion_node(node_pred[masked_node_idx], node_label[masked_node_idx])
        loss_edge = criterion_edge(edge_pred, edge_label) if cfg.train.pretrain.recon == 'all' \
            else criterion_edge(edge_pred[masked_edge_idx], edge_label[masked_edge_idx])
        if hasattr(model.model, 'decode_recon') and callable(model.model.decode_recon):
            node_recon, edge_recon, tuple_recon, node_pe_recon, edge_pe_recon = model.model.decode_recon(pred)
            tuple_label = tuple_label_dict[node_label[batch.edge_index[0]], node_label[batch.edge_index[1]]] #.flatten()

            loss_structure_recon = criterion_node(node_recon, node_label) * cfg.train.pretrain.get('node_factor', 1.0) \
                                 + criterion_edge(edge_recon, edge_label) * cfg.train.pretrain.edge_factor \
                                 + criterion_node(tuple_recon, tuple_label) * cfg.train.pretrain.edge_factor
            if batch.get('pestat_node', None) is not None:
                loss_structure_recon = loss_structure_recon + nn.MSELoss(reduction='mean')(node_pe_recon, batch.get('pestat_node'))
            if batch.get('pestat_edge', None) is not None:
                loss_structure_recon = loss_structure_recon + nn.MSELoss(reduction='mean')(edge_pe_recon, batch.get('pestat_edge'))
        else:
            loss_structure_recon = 0
        loss = loss_node * cfg.train.pretrain.get('node_factor', 1.0) + loss_edge * cfg.train.pretrain.edge_factor
        loss = loss + loss_structure_recon
        if cfg.train.pretrain.recon == 'none':
            loss = 0
        if cfg.train.pretrain.original_task:
            loss_graph, _ = compute_loss(graph_pred, graph_label)
            # loss_graph = criterion_graph(graph_pred, graph_label)
            loss = loss + loss_graph * cfg.train.pretrain.graph_factor
        if cfg.train.pretrain.get('reg', False):
            criterion_reg = nn.MSELoss()
            loss_reg = criterion_reg(pred.x, torch.zeros(pred.x.shape, device=pred.x.device)).mean() + \
                       criterion_reg(pred.edge_attr,
                                     torch.zeros(pred.edge_attr.shape, device=pred.edge_attr.device)).mean()
            loss = loss + loss_reg.mean() * cfg.train.pretrain.get('reg_factor', 0.01)

        # if cfg.dataset.name == 'ogbg-code2':
        #     loss, pred_score = subtoken_cross_entropy(pred, true)
        #     _true = true
        #     _pred = pred_score
        # else:
        #     loss, pred_score = compute_loss(pred, true)
        #     _true = true.detach().to('cpu', non_blocking=True)
        #     _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        _true = graph_label.detach().to('cpu', non_blocking=True)
        _pred = graph_pred.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', repeat=1, ensemble_mode='none'):
    model.eval()
    time_start = time.time()
    tuple_label_dict = create_label_mapping(model.model.node_dict_dim, device=torch.device(cfg.accelerator))

    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            if ensemble_mode == 'none':
                # pred, true = model(batch)
                node_label, edge_label, graph_label = batch.x.clone().detach(), batch.edge_attr.clone().detach(), batch.y
                if cfg.train.pretrain.input_target:
                    batch_1 = copy.deepcopy(batch)
                    input_label = batch_1.y.clone().detach()
                    if cfg.dataset.format == 'PyG-QM9':
                        input_label = (input_label - batch_1.y_mean) / batch_1.y_std
                    pred_ = model(batch_1, label=input_label)
                    node_pred_, edge_pred_, graph_pred_ = model.model.decode(pred_)
                    if cfg.dataset.format == 'PyG-QM9':
                        graph_pred_ = graph_pred_ * batch_1.y_std + batch_1.y_mean
                    loss_labeled, _ = compute_loss(graph_pred_, batch_1.y)
                pred = model(batch, label=None)
                node_pred, edge_pred, graph_pred = model.model.decode(pred)
                if cfg.dataset.format == 'PyG-QM9':
                    graph_pred = graph_pred * batch.get('y_std', 1.) + batch.get('y_mean', 0.)
                if cfg.train.pretrain.recon != 'none':
                    node_label, edge_label = node_label.flatten(), edge_label.flatten()
                    criterion_node = nn.CrossEntropyLoss()
                    criterion_edge = nn.CrossEntropyLoss()
                    loss_node = criterion_node(node_pred, node_label)
                    loss_edge = criterion_edge(edge_pred, edge_label)
                    loss_recon = loss_node * cfg.train.pretrain.get('node_factor', 1.0) + loss_edge * cfg.train.pretrain.edge_factor
                    if hasattr(model.model, 'decode_recon') and callable(model.model.decode_recon):
                        node_recon, edge_recon, tuple_recon, node_pe_recon, edge_pe_recon = model.model.decode_recon(pred)
                        tuple_label = tuple_label_dict[node_label[batch.edge_index[0]], node_label[batch.edge_index[1]]].flatten()
                        loss_structure_recon_node = criterion_node(node_recon, node_label)
                        loss_structure_recon_edge = criterion_edge(edge_recon, edge_label)
                        loss_structure_recon_tuple = criterion_node(tuple_recon, tuple_label)
                        if batch.get('pestat_node', None) is not None:
                            loss_structure_recon_nodepe = nn.MSELoss(reduction='mean')(node_pe_recon, batch.get('pestat_node'))
                        else:
                            loss_structure_recon_nodepe = torch.tensor(0.)
                        if batch.get('pestat_edge', None) is not None:
                            loss_structure_recon_edgepe = nn.MSELoss(reduction='mean')(edge_pe_recon, batch.get('pestat_edge'))
                        else:
                            loss_structure_recon_edgepe = torch.tensor(0.)
                    else:
                        loss_structure_recon_node = torch.tensor(0.)
                        loss_structure_recon_edge = torch.tensor(0.)
                        loss_structure_recon_tuple = torch.tensor(0.)
                        loss_structure_recon_nodepe = torch.tensor(0.)
                        loss_structure_recon_edgepe = torch.tensor(0.)
                else:
                    loss_recon = torch.tensor(0.)
                    loss_structure_recon_node = torch.tensor(0.)
                    loss_structure_recon_edge = torch.tensor(0.)
                    loss_structure_recon_tuple = torch.tensor(0.)
                    loss_structure_recon_nodepe = torch.tensor(0.)
                    loss_structure_recon_edgepe = torch.tensor(0.)
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
            true = batch.y  # TODO: check this
            loss, pred_score = compute_loss(graph_pred, true)
            loss = loss_labeled if cfg.train.pretrain.input_target else loss
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            loss_recon=loss_recon.detach().cpu().item(),
                            loss_recon_node=loss_structure_recon_node.detach().cpu().item(),
                            loss_recon_edge=loss_structure_recon_edge.detach().cpu().item(),
                            loss_recon_tuple=loss_structure_recon_tuple.detach().cpu().item(),
                            loss_recon_nodepe=loss_structure_recon_nodepe.detach().cpu().item(),
                            loss_recon_edgepe=loss_structure_recon_edgepe.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


@register_train('pretrain_encoder')
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
                       cfg.optim.batch_accumulation, cur_epoch % 9 == 0)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1], repeat=cfg.train.ensemble_repeat,
                           ensemble_mode=cfg.train.ensemble_mode)
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
        if cfg.train.enable_ckpt and is_ckpt_epoch(cur_epoch):  # and not cfg.train.ckpt_best:
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


def create_label_mapping(k, device):
    label_mapping = torch.full((k, k), -1, dtype=torch.long, device=device)
    label_counter = 0

    for i in range(k):
        for j in range(i, k):
            label_mapping[i, j] = label_counter
            label_mapping[j, i] = label_counter
            label_counter += 1

    return label_mapping
