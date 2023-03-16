#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/16/2023 1:06 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : deepgmr.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

from lib.o3dutils import reg_solver
from lib.utils import gmm_params
from models.dgcnn import DGCNN, CONV


def gmm_register(pi_s, mu_s, mu_t, sigma_t):
    """
    :param pi_s:
    :param mu_s:
    :param mu_t:
    :param sigma_t:
    :return:
    """
    c_s = pi_s.unsqueeze(1) @ mu_s
    c_t = pi_s.unsqueeze(1) @ mu_t
    Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
                   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1)
    U, _, V = torch.svd(torch.nan_to_num(Ms, nan=0).cpu()+1e-4)
    U = U.cuda()
    V = V.cuda()
    S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    S[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
    R = V @ S @ U.transpose(1, 2)
    t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
    bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
    T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
    return T


class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class DeepGMR(nn.Module):
    def __init__(self, emb_dims, n_clusters, config):
        super().__init__()
        self.backbone = DGCNN(emb_dims, config.gnn_k)
        self.config = config
        self.cluster = CONV(in_size=emb_dims, out_size=n_clusters, hidden_size=emb_dims // 2, used='proj')

    def forward(self, src, tgt, is_test=False):
        batch_size, _, _ = src.size()
        src_feats = self.backbone(src)
        tgt_feats = self.backbone(tgt)

        src_log_scores = self.cluster(src_feats)
        tgt_log_scores = self.cluster(tgt_feats)
        src_gamma = F.softmax(src_log_scores, dim=1)  # [b,k,n]
        tgt_gamma = F.softmax(tgt_log_scores, dim=1)
        src_pi, src_mu, src_sigma = gmm_params(src_gamma.transpose(-1, -2), src.transpose(-1, -2), True)
        tgt_pi, tgt_mu, tgt_sigma = gmm_params(tgt_gamma.transpose(-1, -2), tgt.transpose(-1, -2), True)
        tsfm = gmm_register(src_pi, src_mu, tgt_mu, tgt_sigma)
        if is_test:
            rot, trans = reg_solver(src, tgt, voxel_size=self.config.overlap_radius, trans_init=tsfm)
            return rot, trans
        return tsfm[:, 0:3, 0:3], tsfm[:, 3, 0:3]
