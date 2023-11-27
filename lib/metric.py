#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/14/2022 11:42 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : metric.py
# @Software: PyCharm

import math
import os
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import DataParallel

from datasets.datautils import transform


def concatenate(a: np.ndarray, b: np.ndarray):
    """ Concatenate two SE3 transforms

    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)

    Returns:
        a*b ([B, ] 3/4, 4)

    """

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]
    concatenated = np.concatenate([r_ab, t_ab], axis=-1)
    if a.shape[-2] == 4:
        concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return concatenated


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.model):
        os.makedirs('checkpoints/' + args.model)
    if not os.path.exists('checkpoints/' + args.model + '/' + 'models'):
        os.makedirs('checkpoints/' + args.model + '/' + 'models')


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def stats_error(est, gt, pts):
    r_err = rotation_error(est[:, :3, :3], gt[:, :3, :3]).mean().item()
    t_err = translation_error(est[:, :3, 3], gt[:, :3, 3]).mean().item()
    rmse_err = rmse(pts, est, gt).mean().item()
    return r_err, t_err, rmse_err


def recall(est, gt, r_thresh, t_thresh, eps=1e-6):
    if est is None:
        return np.array([0, np.inf, np.inf])
    rte = np.linalg.norm(est[:3, 3] - gt[:3, 3])
    rre = np.arccos(
        np.clip((np.trace(est[:3, :3].T @ gt[:3, :3]) - 1) / 2,
                -1 + eps, 1 - eps)) * 180 / np.pi
    return np.array([rte < t_thresh and rre < r_thresh, rte, rre])


def rotation_mat2angle(rot):
    return torch.acos(torch.clamp((torch.trace(rot) - 1) / 2, -1.0, 1.0))


def rotation_error(rot1, rot2):
    assert rot1.shape == rot2.shape
    cos_theta = torch.einsum('bij,bij->b', rot1, rot2)
    return torch.arccos(torch.clamp((cos_theta - 1) / 2, -1.0, 1.0)) * 180 / math.pi


def translation_error(t1, t2):
    assert t1.shape == t2.shape
    return torch.norm(t1 - t2, dim=1)


def rmse(pts, est, gt):
    pts_pred = pts @ est[:, :3, :3].transpose(1, 2) + est[:, :3, 3].unsqueeze(1)
    pts_gt = pts @ gt[:, :3, :3].transpose(1, 2) + gt[:, :3, 3].unsqueeze(1)
    return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)


def batch_rotation_error(rots1, rots2):
    r"""
    arccos( (tr(R_1^T R_2) - 1) / 2 )
    rots1: B src_xyz 3 src_xyz 3 or B src_xyz 9
    rots1: B src_xyz 3 src_xyz 3 or B src_xyz 9
    """
    assert len(rots1) == len(rots2)
    trace = (rots1.reshape(-1, 9) * rots2.reshape(-1, 9)).sum(1)
    side = (trace - 1) / 2
    return torch.acos(torch.clamp(side, min=-0.999, max=0.999))


def batch_translation_error(trans1, trans2):
    r"""
    trans1: B src_xyz 3
    trans2: B src_xyz 3
    """
    assert len(trans1) == len(trans2)
    return torch.norm(trans1 - trans2, p=2, dim=1, keepdim=False)


def eval_metrics(output, target):
    output = (torch.sigmoid(output) > 0.5)
    target = target
    return torch.norm(output - target)


def corr_dist(est, gth, xyz, weight=None, max_dist=1):
    xyz_est = xyz @ est[:3, :3].t() + est[:3, 3]
    xyz_gth = xyz @ gth[:3, :3].t() + gth[:3, 3]
    dists = torch.clamp(torch.sqrt(((xyz_est - xyz_gth).pow(2)).sum(1)), max=max_dist)
    if weight is not None:
        dists = weight * dists
    return dists.mean()


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


def get_loss_fn(loss):
    if loss == 'corr_dist':
        return corr_dist
    else:
        raise ValueError(f'Loss {loss}, not defined')


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def inverse(rot: np.ndarray, trans: np.ndarray):
    """Returns the inverse of the SE3 transform
    Args:
        trans:
        rot: ([B,] 3/4, 4) transform
    Returns:
        ([B,] 3/4, 4) matrix containing the inverse
    """
    # rot_pred = g[..., :3, :3]  # (3, 3)
    # gamma = g[..., :3, 3]  # (3)
    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    # if g.shape[-2] == 4:
    #     inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def dcp_metrics(src, tgt, rot_gt, transl_gt, rot_pre, transl_pre, r_th=1.0, t_th=0.1):
    rot_pre, transl_pre = rot_pre.detach(), transl_pre.detach()
    rot_gt, transl_gt = rot_gt.detach(), transl_gt.detach()
    r_pre_euler_deg = npmat2euler(rot_pre.cpu().numpy())
    r_gt_euler_deg = npmat2euler(rot_gt.detach().cpu().numpy())
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((transl_gt - transl_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(transl_gt - transl_pre), dim=1)
    # Rotation, translation errors (isotropic)
    concatenated = concatenate(
        inverse(rot_gt.cpu().numpy(), transl_gt.cpu().numpy()),
        np.concatenate([rot_pre.cpu().numpy(), transl_pre.unsqueeze(-1).cpu().numpy()], axis=-1))
    concatenated = torch.from_numpy(concatenated)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)
    # Chamfer distance
    src_pre_trans = torch.from_numpy(
        transform(torch.cat((rot_pre, transl_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                  src.detach().cpu().numpy())).to(src)
    dist_src = torch.min(square_distance(src_pre_trans, tgt), dim=-1)[0]
    dist_ref = torch.min(square_distance(tgt, src_pre_trans), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)
    src_gt_trans = torch.from_numpy(transform(torch.cat((rot_gt, transl_gt[:, :, None]), dim=2).cpu().numpy(),
                                              src.detach().cpu().numpy())).to(src)
    dist_src = torch.min(square_distance(src_pre_trans, src_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)
    n_correct = (r_mae < r_th) * (t_mae.cpu().numpy() < t_th)
    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(src_pre_trans, tgt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(tgt, src_pre_trans)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'n_correct': to_numpy(n_correct),
               'pre_transform': np.concatenate((to_numpy(rot_pre), to_numpy(transl_pre)[:, :, None]), axis=2),
               'gt_transform': np.concatenate((to_numpy(rot_gt), to_numpy(transl_gt)[:, :, None]), axis=2)}

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k] ** 2))
        elif k.endswith('nomean'):
            summarized[k] = metrics[k]
        elif k.endswith('n_correct'):
            summarized[k] = np.mean(metrics[k])
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(summary_metrics: Dict, title: str = 'Metrics'):
    """Prints out formatted metrics to logger"""

    print('=' * (len(title) + 1))
    print(title + ':')

    print('DeepCP metrics:{:.4f}(rot_pred-rmse) | {:.4f}(rot_pred-mae) | {:.4g}(gamma-rmse) | {:.4g}(gamma-mae)'.
          format(summary_metrics['r_rmse'], summary_metrics['r_mae'],
                 summary_metrics['t_rmse'], summary_metrics['t_mae'],
                 ))
    print('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.
          format(summary_metrics['err_r_deg_mean'],
                 summary_metrics['err_r_deg_rmse']))
    print('Translation error {:.4g}(mean) | {:.4g}(rmse)'.
          format(summary_metrics['err_t_mean'],
                 summary_metrics['err_t_rmse']))
    print('RPM Chamfer error: {:.7f}(mean-sq)'.
          format(summary_metrics['chamfer_dist']))
    print('Source error: {:.7f}(mean-sq)'.
          format(summary_metrics['pcab_dist']))
    print('Clip Chamfer error: {:.7f}(mean-sq)'.
          format(summary_metrics['clip_chamfer_dist']))
    print('Recall: {:.7f}(mean-sq)'.
          format(summary_metrics['n_correct']))


def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module

    torch.save(model.state_dict(), path)
