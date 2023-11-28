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
import numpy as np

from lib.se3 import torch_transform
from lib.utils import gmm_params, contrastsk, get_local_corrs, sinkhorn


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
    def __init__(self, top_k=256):
        super().__init__()
        self.top_k = top_k

    def forward(self, pts, log_gamma, prob=None):
        if prob is not None:
            topk_ids = torch.topk(prob, k=self.top_k, dim=-1)[1].unsqueeze(dim=-1)
            log_score = torch.gather(log_gamma, index=topk_ids.expand(-1, -1, log_gamma.shape[-1]), dim=1)
            pts = torch.gather(pts, index=topk_ids.expand(-1, -1, pts.shape[-1]), dim=1)
            score = torch.softmax(log_score, dim=-1)
        else:
            log_score = log_gamma
            score = torch.softmax(log_gamma, dim=-1)
        pi, mu = gmm_params(score, pts)
        with torch.no_grad():
            # log_score, log_score [B,N,K], p, feats [b,d,k]
            assign, dis = contrastsk(pts, mu, p=pi, max_iter=25, dst='eu')
            assign = assign / assign.sum(dim=-1, keepdim=True).clip(min=1e-4)  # [b, n, k]
        loss = torch.mean(torch.sum(-assign.detach() * torch.log_softmax(log_score, dim=-1), dim=1))
        return loss


class WelschLoss(nn.Module):
    def __init__(self, alpha=1.0, top_k=256):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k

    def forward(self, src, tgt, tsfm, src_feats=None, tgt_feats=None, src_o=None, tgt_o=None):
        if src_feats is not None and tgt_feats is not None:
            if src_o is not None and tgt_o is not None:
                cost = torch.cdist(src_feats, tgt_feats).clip(min=1e-4)
                src_o = src_o / src_o.sum(dim=-1, keepdims=True).clip(min=1e-4)
                tgt_o = tgt_o / tgt_o.sum(dim=-1, keepdims=True).clip(min=1e-4)
                score = sinkhorn(cost, p=src_o, q=tgt_o)[0].clip(min=1e-4)
                score = score / score.sum(dim=-1, keepdims=True).clip(min=1e-4)
            else:
                score = torch.softmax(torch.matmul(
                    src_feats, tgt_feats.transpose(-1, -2)) / np.sqrt(src_feats.shape[-1]), dim=-1)
            # score = torch.softmax(torch.matmul(
            #     src_feats, tgt_feats.transpose(-1, -2)) / np.sqrt(src_feats.shape[-1]), dim=-1)
            tgt_corr = torch.matmul(score, tgt)
            # ids = torch.topk(score.max(dim=-1)[0], k=self.top_k, dim=-1)[1].detach()
            # ids = ids.unsqueeze(-1).expand(-1, -1, src.size(-1))
            # src = torch.gather(src, index=ids, dim=1)
            # tgt_corr = torch.gather(tgt_corr, index=ids, dim=1)
        else:
            tgt_corr = tgt
        src = torch_transform(tsfm, src, normals=None)
        alpha2 = self.alpha * self.alpha
        z = torch.norm(src - tgt_corr, dim=-1)
        loss = (1.0 - torch.exp(-0.5 * torch.pow(z, 2.0) / alpha2)).sum(dim=-1)
        return loss.mean()


class CluLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.loss = ConLoss(tau)

    def forward(self, xyz, xyz_mu, feats, gamma):
        feats_pos = gmm_params(gamma, feats.transpose(-1, -2))[1]
        feats_anchor = get_local_corrs(xyz.transpose(-1, -2), xyz_mu.transpose(-1, -2), feats.transpose(-1, -2))
        loss = self.loss(feats_anchor, feats_pos)
        return loss


def dcp_loss(rot_pred, rot_gt, transl_pred, transl_gt):
    batch_size = transl_gt.shape[0]
    transl_gt, transl_pred = transl_gt.view(batch_size, 3), transl_pred.view(batch_size, 3)
    identity = torch.eye(3).to(rot_pred).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = F.mse_loss(torch.matmul(rot_pred.transpose(2, 1), rot_gt), identity) + F.mse_loss(transl_pred, transl_gt)
    return loss


def con_loss(src_desc, tgt_desc, tau=0.1):
    conloss = ConLoss(tau)
    src_desc = src_desc.transpose(-1, -2)
    tgt_desc = tgt_desc.transpose(-1, -2)
    loss = conloss(src_desc, tgt_desc)
    return loss


def get_weighted_bce_loss(prediction, gt):

    return F.mse_loss(prediction, gt)
