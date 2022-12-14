#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/2022 11:06 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : dgcnn.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from lib.se3 import compute_rigid_transformation
from lib.utils import knn, get_graph_feature, sinkhorn, cos_similarity, og_params


class CONV(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )

    def forward(self, x):
        return self.net(x)


class GMMSVD(nn.Module):
    def __init__(self, is_sk=True):
        super().__init__()
        self.is_sk = is_sk

    def forward(self, src, tgt, src_desc, tgt_desc, src_log_gamma=None, tgt_log_gamma=None, src_o=None, tgt_o=None):
        src_gamma = F.softmax(src_log_gamma, dim=1)  # [b,k,n]
        tgt_gamma = F.softmax(tgt_log_gamma, dim=1)
        src_pi, src_mu, src_desc_mu = og_params(src.transpose(-1, -2), src_gamma.transpose(-1, -2), src_o, src_desc)
        tgt_pi, tgt_mu, tgt_desc_mu = og_params(tgt.transpose(-1, -2), tgt_gamma.transpose(-1, -2), tgt_o, tgt_desc)
        src_desc_l = src_desc_mu[:, :, -1].detach()
        tgt_desc_l = tgt_desc_mu[:, :, -1].detach()
        src_desc_mu[:, :, -1] = tgt_desc_l
        tgt_desc_mu[:, :, -1] = src_desc_l
        ############################################################
        batch_size, num_points_tgt, _ = tgt_mu.size()
        batch_size, num_points, _ = src_desc_mu.size()
        similarity = cos_similarity(src_desc_mu, tgt_desc_mu)  # [b, n+1, m+1]
        # point-wise matching map
        if self.is_sk:
            cost = 1.0 - similarity
            scores = sinkhorn(cost, p=src_pi, q=tgt_pi, epsilon=1e-2, thresh=1e-2, max_iter=30)[0]
            src_scores = scores[:, :-1, :-1]
            pi = torch.sum(src_scores, dim=-1, keepdim=True).clip(min=1e-4)
            src_scores = src_scores / pi
        else:
            src_scores = torch.softmax(similarity[:, :-1, :-1] / 0.05, dim=2)  # [b, n, m]  Eq. (1)

        src_corr = torch.einsum('bmd,bnm->bdn', tgt_mu[:, :-1, :], src_scores)  # [b,d,n] Eq. (4)
        # compute rigid transformation
        weight = src_scores.sum(dim=-1).unsqueeze(1)
        R, t = compute_rigid_transformation(src_mu[:, :-1, :].transpose(-1, -2), src_corr, weight)

        return R, t.view(batch_size, 3), src_corr.transpose(-1, -2), src_scores


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512, k=20):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
        self.k = k

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        idx = knn(x.transpose(-1, -2), x.transpose(-1, -2), k=self.k)

        x = get_graph_feature(x, k=self.k, idx=idx)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv5(x))).view(
            batch_size, -1, num_points)

        return x
