import logging
import time

import numpy as np
import random
import torch
import torch.nn as nn
from fcd_torch import FCD as FCDMetric
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
from lgd.asset.molecules_evaluation import compute_molecular_metrics, build_molecule, mols_to_nx, mol2smiles, build_molecule_with_partial_charges
from lgd.asset.mmd import compute_nspdk_mmd
from lgd.asset.stats import eval_graph_list
from rdkit import Chem
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    torch.autograd.set_detect_anomaly(True)
    generated_graph = []
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        # TODO: for masked_graph condition, should have masked batch for cond_stage_model, and unmasked batch for first_stage_model
        #  hence need to use ground truth for batch.x and batch.edge_attr, and mask batch.x_masked and batch.edge_attr_masked

        node_label, edge_label, graph_label = batch.x.clone().detach().flatten(), batch.edge_attr.clone().detach().flatten(), batch.y
        # the embed of labels and prefix are done in fine-tuning of the encoder, not pretraining
        batch.x_masked = batch.x.clone().detach()
        batch.edge_attr_masked = batch.edge_attr.clone().detach()
        loss, loss_task, graph_pred, loss_node, loss_edge, loss_graph, loss_encoder = model.training_step(batch)
        # logging.info(len(graph_pred))
        for each in graph_pred:
            generated_graph.append(each)
        if cfg.diffusion.cond_stage_key != 'unconditional':
            loss = loss + loss_task * cfg.diffusion.get("task_factor", 0.0)
        # with torch.autograd.detect_anomaly():
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                # TODO: gradient already have nan?
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        # for qm9 unconditional generation, we do not care about MAE
        _true = graph_label.detach().to('cpu', non_blocking=True)
        _pred = graph_label.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            loss_node=loss_node.detach().cpu().item(),
                            loss_edge=loss_edge.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
    # validity_dict, rdkit_metrics, all_smiles = compute_molecular_metrics(generated_mol, train_smiles, dataset_info)


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', repeat=1, ensemble_mode='none', evaluate=False, current_epoch=0, test_graph_list=None, methods=None, kernels=None):
    model.eval()
    time_start = time.time()
    generated_graph = []
    iter = 0
    for batch in loader:
        if iter == 0 and evaluate:
            visualize = True
            iter += 1
        else:
            visualize = False
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            if ensemble_mode == 'none':
                node_label, edge_label, graph_label = batch.x.clone().detach().flatten(), batch.edge_attr.clone().detach().flatten(), batch.y
                # the embed of labels and prefix are done in fine-tuning of the encoder, not pretraining
                batch.x_masked = batch.x.clone().detach()
                batch.edge_attr_masked = batch.edge_attr.clone().detach()
                ddim_steps = cfg.diffusion.get('ddim_steps', None)
                ddim_eta = cfg.diffusion.get('ddim_eta', 0.0)
                use_ddpm_steps = cfg.diffusion.get('use_ddpm_steps', False)
                _, graph_pred = model.inference(batch, ddim_steps=ddim_steps, ddim_eta=ddim_eta, use_ddpm_steps=use_ddpm_steps, visualize=visualize)
                for each in graph_pred:
                    generated_graph.append(each)
                # logging.info('graph_pred')
                # logging.info(graph_pred)
                # pred, true = model(batch)
                # pred = model(batch)
                # node_pred, edge_pred, graph_pred = model.model.decode(pred)
            else:
                raise NotImplementedError
                # batch_pred = []
                # for i in range(repeat):
                #     bc = deepcopy(batch)
                #     bc.x_masked = batch.x.clone().detach()
                #     bc.edge_attr_masked = batch.edge_attr.clone().detach()
                #     loss_generation, loss_graph, graph_pred = model.validation_step(bc)
                #     batch_pred.append(graph_pred)
                #     del bc
                # batch_pred = torch.cat(batch_pred).reshape(repeat, -1)
                # if ensemble_mode == 'mean':
                #     graph_pred = torch.mean(batch_pred, dim=0)
                # else:
                #     graph_pred = torch.median(batch_pred, dim=0)[0]
            # pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            true = batch.y  # TODO: check this
            # loss, pred_score = compute_loss(graph_pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = true.detach().to('cpu', non_blocking=True)
            # logging.info(_pred)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=_.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    if evaluate:
        result_dict = eval_graph_list(test_graph_list, generated_graph, methods=methods, kernels=kernels)
        # visualize generated molecules
        visualize_samples = random.sample(generated_graph, 5)
        save_path = os.path.join(cfg.run_dir, 'generated_graphs')
        os.makedirs(save_path, exist_ok=True)

        for i in range(len(visualize_samples)):
            G = visualize_samples[i]
            logging.info(G)
            labels = nx.get_node_attributes(G, 'label')
            pos = nx.spring_layout(G)
            nx.draw(G, pos)
            nx.draw_networkx_labels(G, pos, labels=labels)
            nx.draw_networkx_edge_labels(G, pos)
            plt.savefig(save_path + '/epoch_' + str(current_epoch) + '_sample_' + str(i) + '.png')
            plt.clf()


@register_train('generic_generation')
def custom_train_diffusion(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: diffusion model
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
    train_loader, test_loader = loaders[0], loaders[2]
    generic_graphs = []
    for i, batch in enumerate(test_loader):
        accumulated_node, accumulated_edge = 0, 0
        for i in range(batch.num_graphs):
            adj = batch.edge_attr[accumulated_edge: accumulated_edge + batch.num_node_per_graph[i] ** 2] \
                .reshape(batch.num_node_per_graph[i], batch.num_node_per_graph[i]).detach().cpu().numpy()
            G = nx.from_numpy_array(adj)
            G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() < 1:
                G.add_node(1)
            generic_graphs.append(G)
            accumulated_node += batch.num_node_per_graph[i]
            accumulated_edge += batch.num_node_per_graph[i] ** 2

    # visualize test graphs
    save_path = os.path.join(cfg.run_dir, 'generated_graphs')
    os.makedirs(save_path, exist_ok=True)
    for i in range(5):
        G = generic_graphs[i]
        logging.info(G)
        labels = nx.get_node_attributes(G, 'label')
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        nx.draw_networkx_labels(G, pos, labels=labels)
        nx.draw_networkx_edge_labels(G, pos)
        plt.savefig(save_path + '/test_ground_truth_' + str(i) + '.png')
        plt.clf()

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        if cur_epoch in [999, 1999, 2499]:
            cfg.train.eval_period = int(cfg.train.eval_period / 2)
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if cur_epoch > cfg.train.start_eval_epoch:
            if is_eval_epoch(cur_epoch):
                if cur_epoch == 0:
                    eval_epoch(loggers[1], loaders[1], model,
                               split=split_names[0], repeat=cfg.train.ensemble_repeat,
                               ensemble_mode=cfg.train.ensemble_mode,
                               evaluate=True, current_epoch=cur_epoch, test_graph_list=generic_graphs, methods=cfg.dataset.methods, kernels=cfg.dataset.kernels)
                    perf[1].append(loggers[1].write_epoch(cur_epoch))
                else:
                    perf[1].append(perf[1][-1])

                eval_epoch(loggers[2], loaders[2], model,
                           split=split_names[1], repeat=cfg.train.ensemble_repeat, ensemble_mode=cfg.train.ensemble_mode,
                           evaluate=True, current_epoch=cur_epoch, test_graph_list=generic_graphs, methods=cfg.dataset.methods, kernels=cfg.dataset.kernels)
                perf[2].append(loggers[2].write_epoch(cur_epoch))
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
            if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                    and (is_ckpt_epoch(cur_epoch) or ((cur_epoch + 1) % 100 == 0)):
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()

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
