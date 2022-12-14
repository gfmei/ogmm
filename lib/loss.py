#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/2022 11:13 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from lib.utils import wkeans, gmm_params, contrastsk, get_local_corrs


class ConLoss(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tau = tau

    def forward(self, x, y, normalize=True):
        """
        :param normalize:
        :param x: (bs, n, d)
        :param y: (bs, n, d)
        :return: loss
        """
        bs, n, dim = y.shape
        labels = torch.zeros((2 * n,), dtype=torch.long, device=x.device).expand(bs, -1).reshape(-1)
        mask = torch.ones((n, n), dtype=bool, device=x.device).fill_diagonal_(0).expand(bs, -1, -1)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        score_xy = torch.einsum('bmd,bnd->bmn', x, y) / self.tau
        score_yx = torch.einsum('bmd,bnd->bmn', y, x) / self.tau
        score_xx = torch.einsum('bmd,bnd->bmn', x, x) / self.tau
        score_yy = torch.einsum('bmd,bnd->bmn', y, y) / self.tau
        # Compute Positive Logits
        logits_xy_pos = score_xy[torch.logical_not(mask)].view(bs, -1)
        logits_yx_pos = score_yx[torch.logical_not(mask)].view(bs, -1)
        # Compute Negative Logits
        logit_xx_neg = score_xx[mask].reshape(bs, n, -1)
        logit_yy_neg = score_yy[mask].reshape(bs, n, -1)
        logit_xy_neg = score_xy[mask].reshape(bs, n, -1)
        logit_yx_neg = score_yx[mask].reshape(bs, n, -1)
        # Postive Logits over all samples
        pos = torch.cat((logits_xy_pos, logits_yx_pos), dim=1).unsqueeze(-1)
        # Negative Logits over all samples
        neg_x = torch.cat((logit_xx_neg, logit_xy_neg), dim=2)
        neg_y = torch.cat((logit_yx_neg, logit_yy_neg), dim=2)
        neg = torch.cat((neg_x, neg_y), dim=1)
        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=2).view(-1, 2 * n - 1)
        loss = self.ce(logits, labels)

        return loss


class KMLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.loss = ConLoss(tau)

    def forward(self, pts, feats, num_clusters):
        gamma = wkeans(feats, num_clusters)[0]
        pi, mu = gmm_params(gamma, pts)
        with torch.no_grad():
            # log_score, log_score [B,N,K], p, feats [b,d,k]
            assign, dis = contrastsk(pts, mu, max_iter=25, dst='eu')
            assign = assign / assign.sum(dim=-1, keepdim=True).clip(min=1e-4)  # [b, n, k]
        loss = torch.mean(torch.sum(-(assign.detach() * torch.log(gamma.clip(min=1e-3))).clip(max=1.0), dim=1))
        return loss


class CluLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.loss = ConLoss(tau)

    def forward(self, xyz, xyz_mu, feats, gamma):
        feats_pos = gmm_params(gamma, feats.transpose(-1, -2))[1]
        feats_anchor = get_local_corrs(xyz.transpose(-1, -2), xyz_mu.transpose(-1, -2), feats.transpose(-1, -2))
        loss = self.loss(feats_anchor, feats_pos)
        return loss
