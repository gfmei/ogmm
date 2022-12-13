#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2022 10:52 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : gmmreg.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

from lib.loss import CluLoss
from lib.o3dutils import integrate_trans, reg_solver
from lib.utils import get_anchor_corrs, gmm_params
from models.attn import Transformer, PositionEncoding
from models.dgcnn import DGCNN, CONV, GMMSVD


class GMMReg(nn.Module):
    def __init__(self, emb_dims, n_clusters, config):
        super().__init__()
        self.emd = DGCNN(emb_dims, config.gnn_k)
        self.overlap = CONV(in_size=self.emb_dims + 1, out_size=1, hidden_size=self.emb_dims // 2, used='proj')
        self.n_clusters = n_clusters
        self.cluster = CONV(in_size=self.emb_dims, out_size=n_clusters, hidden_size=self.emb_dims // 2, used='proj')
        self.soft_svd = GMMSVD(False)
        self.pos = PositionEncoding(emb_dims)
        self.sattn = Transformer(emb_dims, config.num_heads)
        self.cattn = Transformer(emb_dims, config.num_heads)
        self.cluloss = CluLoss(tau=0.1)

    def forward(self, src, tgt, is_test=False):
        batch_size, _, _ = src.size()
        # rot = torch.eye(3, device=src_mu.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        # trans = torch.zeros(batch_size, 3, 1, device=src_mu.device, dtype=torch.float32)
        # loss = torch.tensor(0.0).to(src_mu.device)
        src_feats = self.emd(src)
        tgt_feats = self.emd(tgt)
        src_feats_anchor, src_feats_pos, src_gamma, src_pi, src_xyz_mu = get_anchor_corrs(
            src, src_feats, self.n_clusters, dst='eu', iters=10, is_fast=True)
        tgt_feats_anchor, tgt_feats_pos, tgt_gamma, tgt_pi, tgt_xyz_mu = get_anchor_corrs(
            tgt, tgt_feats, self.n_clusters, dst='eu', iters=10, is_fast=True)
        src_pos = self.pos(src, 5)
        tgt_pos = self.pos(tgt, 5)
        src_feats_t = src_feats + src_pos
        tgt_feats_t = tgt_feats + tgt_pos
        src_feats_t = self.sattn(src_feats_t, src_feats_pos)
        tgt_feats_t = self.sattn(tgt_feats_t, tgt_feats_pos)
        # feats_anchor = get_local_corrs(xyz, xyz_mu, feats).transpose(-1, -2)
        src_feats_pos = gmm_params(src_gamma, src_feats_t.transpose(-1, -2))[1].transpose(-1, -2)
        tgt_feats_pos = gmm_params(tgt_gamma, tgt_feats_t.transpose(-1, -2))[1].transpose(-1, -2)
        src_feats_t = self.sattn(src_feats_t, tgt_feats_pos)
        tgt_feats_t = self.cattn(tgt_feats_t, src_feats_pos)
        src_feats = src_feats + src_feats_t
        tgt_feats = tgt_feats + tgt_feats_t

        # compute overlap scores
        src_fn, tgt_fn = F.normalize(src_feats), F.normalize(tgt_feats)
        similarity = torch.einsum('bdm,bdn->bmn', src_fn, tgt_fn)
        src_co = similarity.max(dim=2)[0].unsqueeze(1)
        tgt_co = similarity.max(dim=1)[0].unsqueeze(1)
        # [B, 1, N]
        src_o = torch.sigmoid(self.overlap(torch.cat([src_feats, src_co], dim=1)) / 0.1)
        tgt_o = torch.sigmoid(self.overlap(torch.cat([tgt_feats, tgt_co], dim=1)) / 0.1)

        src_log_scores = self.cluster(src_feats)
        tgt_log_scores = self.cluster(tgt_feats)

        src_clu_loss = self.cluloss(src, src_xyz_mu, src_feats, src_gamma)[1]
        tgt_clu_loss = self.cluloss(tgt, tgt_xyz_mu, tgt_feats, tgt_gamma)[1]
        # rot, trans, soft_tgt, soft_src_scores, soft_src, soft_tgt_scores = self.soft_svd(
        # src_mu, tgt_mu, src_feats, tgt_feats)
        rot, trans, soft_tgt_mu, soft_src_mu_scores, soft_src_mu, soft_tgt_mu_scores = self.soft_svd(
            src, tgt, src_feats, tgt_feats, src_log_scores, tgt_log_scores)
        # iv_transf = soft_svd(tgt_mu_xyz, src_mu_xyz, tgt_mu_feats, src_mu_feats)
        # self.crsloss(tgt_mu, src_mu, tgt_log_scores, src_log_scores, iv_transf[0], iv_transf[1])
        clu_loss = 0.5*(src_clu_loss + tgt_clu_loss)
        # we_loss = self.we_loss(tgt_mu_xyz, soft_tgt_mu)
        # + self.we_loss(
        #     src_mu, soft_src, src_scores.transpose(-1, -2))
        # st_loss = gw_loss(src_mu_feats, tgt_mu_feats, src_mu_xyz, tgt_mu_xyz, src_pi, tgt_pi).clip(min=0.0)
        loss = clu_loss
        if is_test:
            trans_init = integrate_trans(rot, trans)
            rot, trans = reg_solver(src, tgt, voxel_size=0.02, trans_init=trans_init)

        return rot, trans, src_o, tgt_o, loss