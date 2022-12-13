#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/2022 11:41 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : attn.py
# @Software: PyCharm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import get_graph_feature


def MLP(channels: list, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
                layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class PositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_dis = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv_ang1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv_ang2 = nn.Sequential(
            nn.Conv1d(64, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        l_dim = dim // 2 + dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(l_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, points, k=5):
        """
        :param k: The number of neighbors
        :param points: [B, dim, N]
        :return:
        """
        centroid = torch.mean(points, dim=-1, keepdim=True)  # [B, dim, 1]
        p2gc = points - centroid  # [B, dim, num]
        g_dis = torch.square(p2gc).sum(dim=1, keepdim=True)
        dis_feature = self.conv_dis(g_dis)
        p2lc = get_graph_feature(points, k, idx=None, extra_dim=False)[:, :3, :, :]  # [B, dim, num, k]
        p2gc_n = F.normalize(p2gc, dim=1)
        p2lc_n = F.normalize(p2lc, dim=1)
        alpha = torch.einsum('bdnk,bdn->bnk', p2lc_n, p2gc_n).unsqueeze(1)
        ang_feature = self.conv_ang2(self.conv_ang1(alpha).max(dim=-1, keepdim=False)[0])
        feature = torch.cat([dis_feature, ang_feature], dim=1)
        return feature


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class Transformer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, src, tgt):
        message = self.attn(src, tgt, tgt)
        return self.mlp(torch.cat([src, message], dim=1))
