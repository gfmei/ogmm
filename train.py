#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/14/2022 11:35 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : train.py
# @Software: PyCharm

import logging
import os
from collections import defaultdict
from time import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from configs.cfgs import mnet
from datasets.dataloader import data_loader
from lib.loss import dcp_loss, get_weighted_bce_loss
from lib.metric import rotation_error, translation_error, dcp_metrics, summarize_metrics, save_model, _init_
from lib.se3 import decompose_trans
from models.gmmreg import GMMReg


def train_one_epoch(epoch, model, loader, optimizer, logger, checkpoint_path):
    model.train()
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    all_train_metrics_np = defaultdict(list)

    logger.info("===========================Training: Epoch {:<4}==============================".format(epoch))
    start = time()
    for step, data in enumerate(tqdm(loader, leave=False)):
        pts1 = data["src_xyz"]
        pts2 = data["tgt_xyz"]
        src_overlap = data["src_overlap"]
        tgt_overlap = data["tgt_overlap"]
        transf_gt = data["transform_gt"]
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            src_overlap = src_overlap.cuda().squeeze(-1)
            tgt_overlap = tgt_overlap.cuda().squeeze(-1)
            transf_gt = transf_gt.cuda()
        pts1 = pts1.transpose(-1, -2)
        pts2 = pts2.transpose(-1, -2)
        data_time.append(time() - start)
        optimizer.zero_grad()
        rot_gt, trans_gt = decompose_trans(transf_gt)
        batch_size = transf_gt.shape[0]
        trans_gt = trans_gt.view(batch_size, 3)
        rot, trans, src_o, tgt_o, clu_loss = model(pts1, pts2)
        o_pred = torch.cat([src_o, tgt_o], dim=-1)
        o_gt = torch.cat([src_overlap, tgt_overlap], dim=-1)
        o_pred, o_gt = torch.nan_to_num(o_pred, nan=0.0).clip(min=0.0), torch.nan_to_num(o_gt, nan=0.0).clip(min=0.0)
        assert o_pred.shape == o_gt.shape
        if torch.cuda.device_count() > 1:
            clu_loss = clu_loss.sum()
        r_err = rotation_error(rot, rot_gt)
        t_err = translation_error(trans, trans_gt)
        try:
            loss = dcp_loss(rot, rot_gt, trans, trans_gt) + clu_loss + get_weighted_bce_loss(o_pred, o_gt)
            loss = torch.nan_to_num(loss, nan=0.0)
        except Exception as e:
            loss = dcp_loss(rot, rot_gt, trans, trans_gt) + clu_loss
        loss.backward()
        optimizer.step()
        batch_time.append(time() - start)
        # training accuracy statistic
        losses.append(loss.item())
        r_errs.append(r_err.mean().item())
        t_errs.append(t_err.mean().item())
        src, tgt = pts1[:, :3, :], pts2[:, :3, :]
        perform_metrics = dcp_metrics(src.transpose(-1, -2), tgt.transpose(-1, -2), rot_gt, trans_gt, rot, trans)
        for k in perform_metrics:
            all_train_metrics_np[k].append(perform_metrics[k])
        all_train_metrics_np['loss'].append(np.repeat(clu_loss.item(), 4))

    all_train_metrics_np = {k: np.concatenate(all_train_metrics_np[k]) for k in all_train_metrics_np}
    summary_metrics = summarize_metrics(all_train_metrics_np)
    # print_metrics(summary_metrics)
    logger.info(
        'Epoch {:<4} Mean-Loss: {:.4f} r_rmse:{:.4f} r_mae:{:.4f} t_rmse:{:.4f} t_mae:{:.4f} '
        'clip_dist:{:.4f} recall:{:.4f}'.format(
            epoch, summary_metrics['loss'], summary_metrics['r_rmse'], summary_metrics['r_mae'],
            summary_metrics['t_rmse'], summary_metrics['t_mae'], summary_metrics['clip_chamfer_dist'],
            summary_metrics['n_correct']))
    if (epoch + 1) % 50 == 0:
        model_path_i = os.path.join(checkpoint_path, 'models/model_{:04}.pt'.format(epoch + 1))
        save_model(model, model_path_i)
    batch_time.clear()
    data_time.clear()
    losses.clear()
    r_errs.clear()
    t_errs.clear()
    return summary_metrics


def eval_one_epoch(epoch, model, loader, logger):
    model.eval()
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    all_val_metrics_np = defaultdict(list)
    logger.info("===========================Test: Epoch {:<4}==============================".format(epoch))
    start = time()
    for step, data in enumerate(tqdm(loader, leave=False)):
        pts1 = data["src_xyz"]
        pts2 = data["tgt_xyz"]
        src_overlap = data["src_overlap"]
        tgt_overlap = data["tgt_overlap"]
        transf_gt = data["transform_gt"]
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            src_overlap = src_overlap.cuda().squeeze(-1)
            tgt_overlap = tgt_overlap.cuda().squeeze(-1)
            transf_gt = transf_gt.cuda()
        pts1 = pts1.transpose(-1, -2)
        pts2 = pts2.transpose(-1, -2)
        rot_gt, trans_gt = decompose_trans(transf_gt)
        data_time.append(time() - start)
        with torch.no_grad():
            batch_size = transf_gt.shape[0]
            trans_gt = trans_gt.view(batch_size, 3)
            rot, trans, src_o, tgt_o, clu_loss = model(pts1, pts2, True)
            batch_time.append(time() - start)
            o_pred = torch.cat([src_o, tgt_o], dim=-1)
            o_gt = torch.cat([src_overlap, tgt_overlap], dim=-1)
            if torch.cuda.device_count() > 1:
                clu_loss = clu_loss.sum()
            r_err = rotation_error(rot, rot_gt)
            t_err = translation_error(trans, trans_gt)
            try:
                loss = dcp_loss(rot, rot_gt, trans, trans_gt) + clu_loss + get_weighted_bce_loss(o_pred, o_gt)
            except Exception as e:
                loss = dcp_loss(rot, rot_gt, trans, trans_gt) + clu_loss
            # training accuracy statistic
            losses.append(loss.item())
            r_errs.append(r_err.mean().item())
            t_errs.append(t_err.mean().item())
            src, tgt = pts1[:, :3, :], pts2[:, :3, :]
            perform_metrics = dcp_metrics(src.transpose(-1, -2), tgt.transpose(-1, -2), rot_gt, trans_gt, rot, trans)
        batch_cur_size = src.size(0)
        for k in perform_metrics:
            all_val_metrics_np[k].append(perform_metrics[k])
            all_val_metrics_np['loss'].append(np.repeat(loss.item(), 4))

        all_val_metrics_np['loss'].append(np.repeat(loss.item(), batch_cur_size))

    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    summary_metrics = summarize_metrics(all_val_metrics_np)
    # print_metrics(summary_metrics)
    logger.info(
        'Epoch {:<4} Mean-Loss: {:.4f} r_rmse:{:.4f} r_mae:{:.4f} t_rmse:{:.4f} t_mae:{:.4f} '
        'clip_dist:{:.4f} recall:{:.4f}'.format(
            epoch, summary_metrics['loss'], summary_metrics['r_rmse'], summary_metrics['r_mae'],
            summary_metrics['t_rmse'], summary_metrics['t_mae'], summary_metrics['clip_chamfer_dist'],
            summary_metrics['n_correct']))

    return summary_metrics


if __name__ == "__main__":
    args = mnet()
    # args = indoor()
    _init_(args)
    if args.model == 'GMMReg':
        model = GMMReg(args.emb_dims, args.n_clusters, args)
    else:
        raise Exception('Not implemented')
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # fetch dataloaders
    train_loader, test_loader = data_loader(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 200], gamma=0.1)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    logfile = os.path.join(args.model_path, 'checkpoints/{}/train.log'.format(args.exp_name))
    handler = logging.FileHandler(logfile, encoding='UTF-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    optimal_rot = np.inf
    optimal_tra = np.inf
    optimal_ccd = np.inf
    optimal_recall = -np.inf
    checkpoint_path = os.path.join(args.model_path, 'checkpoints/{}'.format(args.exp_name))
    optim_path = os.path.join(checkpoint_path, 'models/optim_model.pt')
    if os.path.exists(optim_path):
        try:
            logger.info('Loading optimizer state from {}'.format(optim_path))
            model.load_state_dict(torch.load(optim_path))
        except Exception as e:
            model.module.load_state_dict(torch.load(optim_path))
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, logger, checkpoint_path)
        val_metrics = eval_one_epoch(epoch, model, test_loader, logger)
        if optimal_recall < val_metrics['n_correct']:
            optimal_rot = val_metrics['r_mae']
            optimal_tra = val_metrics['t_mae']
            optimal_ccd = val_metrics['clip_chamfer_dist']
            optimal_recall = val_metrics['n_correct']
            save_model(model, optim_path)
        logger.info('Current best rotation: {:.04f}, transl: {:.04f}, ccd: {:.04f}, recall: {:.04f}'.format(
            optimal_rot, optimal_tra, optimal_ccd, optimal_recall))
        scheduler.step()
    logger.debug('train, end')
    logger.debug('done (PID=%d)', os.getpid())
