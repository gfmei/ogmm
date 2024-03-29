#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2022 10:52 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : gmmreg.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from lib.loss import CluLoss
from lib.o3dutils import integrate_trans, reg_solver
from lib.utils import get_anchor_corrs, wkeans_plus
from models.attn import Transformer, PositionEncoding
from models.dgcnn import DGCNN, CONV, GMMSVD


class Clustering(nn.Module):
    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def forward(self, xyz, feats, o_scores):
        # feats = self.cluster(feats)
        xyz = xyz.transpose(-1, -2)
        feats = feats.transpose(-1, -2)
        gamma, pi, node_xyz, node_feats = wkeans_plus(xyz, feats, o_scores, self.n_clusters, iters=10, tau=1.0)
        return gamma, pi, node_xyz, node_feats


class GMMReg(nn.Module):
    def __init__(self, emb_dims, n_clusters, config):
        super().__init__()
        self.emd = DGCNN(emb_dims, config.gnn_k)
        self.proj = CONV(in_size=emb_dims, out_size=1, hidden_size=emb_dims // 2, used=None)
        self.overlap = CONV(in_size=emb_dims, out_size=1, hidden_size=emb_dims // 2, used='proj')
        self.conv1 = CONV(in_size=emb_dims, out_size=emb_dims, hidden_size=2 * emb_dims, used='proj')
        self.conv2 = CONV(in_size=emb_dims + 2, out_size=emb_dims, hidden_size=2 * emb_dims, used='proj')
        self.cluster = Clustering(n_clusters)
        self.soft_svd = GMMSVD(False)
        self.config = config
        self.pos = PositionEncoding(emb_dims)
        self.sattn1 = Transformer(emb_dims, config.num_heads)
        self.cattn = Transformer(emb_dims, config.num_heads)
        self.sattn2 = Transformer(emb_dims, config.num_heads)
        self.cluloss = CluLoss(tau=0.1)
        # self.cluloss = KMLoss(config.n_keypoints)

    def forward(self, src, tgt, is_test=False):
        batch_size, _, _ = src.size()
        src_feats = self.emd(src)
        tgt_feats = self.emd(tgt)
        src_feats_anchor, src_feats_pos, src_xyz_mu = get_anchor_corrs(
            src, src_feats, self.config.km_clusters, dst='eu', iters=10, is_fast=True)
        tgt_feats_anchor, tgt_feats_pos, tgt_xyz_mu = get_anchor_corrs(
            tgt, tgt_feats, self.config.km_clusters, dst='eu', iters=10, is_fast=True)
        src_pos = self.pos(src, 5)
        tgt_pos = self.pos(tgt, 5)
        src_feats_t = src_feats + src_pos
        tgt_feats_t = tgt_feats + tgt_pos
        src_feats_t = self.conv1(self.sattn1(src_feats_t, src_feats_pos) + src_feats_t)
        tgt_feats_t = self.conv1(self.sattn1(tgt_feats_t, tgt_feats_pos) + tgt_feats_t)

        # src_feats_pos = gmm_params(src_gamma, src_feats_t.transpose(-1, -2))[1].transpose(-1, -2)
        # tgt_feats_pos = gmm_params(tgt_gamma, tgt_feats_t.transpose(-1, -2))[1].transpose(-1, -2)
        src_feats_pos = get_anchor_corrs(
            src, src_feats_t, self.config.km_clusters, dst='eu', iters=10, is_fast=True)[1]
        tgt_feats_pos = get_anchor_corrs(
            tgt, tgt_feats_t, self.config.km_clusters, dst='eu', iters=10, is_fast=True)[1]
        src_feats = self.cattn(src_feats_t, tgt_feats_pos) + src_feats_t
        tgt_feats = self.cattn(tgt_feats_t, src_feats_pos) + tgt_feats_t
        # compute overlap scores
        src_fn, tgt_fn = F.normalize(src_feats), F.normalize(tgt_feats)
        similarity = torch.einsum('bdm,bdn->bmn', src_fn, tgt_fn)
        src_o, tgt_o = self.proj(src_feats), self.proj(tgt_feats)
        # src_co = similarity.max(dim=2)[0].unsqueeze(1)
        # tgt_co = similarity.max(dim=1)[0].unsqueeze(1)
        src_wo = torch.einsum('bmn,bdn->bdm', torch.softmax(similarity, dim=-1), src_o)
        tgt_wo = torch.einsum('bmn,bdm->bdn', torch.softmax(similarity, dim=1), tgt_o)
        # [B, 1, N]
        src_feats_o = torch.cat([src_feats, src_wo, src_o], dim=1)
        tgt_feats_o = torch.cat([tgt_feats, tgt_wo, tgt_o], dim=1)
        src_feats_o, tgt_feats_o = self.conv2(src_feats_o), self.conv2(tgt_feats_o)
        src_o, tgt_o = self.overlap(src_feats_o), self.overlap(tgt_feats_o)
        # src_feats, src_o = src_feats_o[:, :-1, :], src_feats_o[:, -1, :]
        # tgt_feats, tgt_o = tgt_feats_o[:, :-1, :], tgt_feats_o[:, -1, :]
        src_o = torch.sigmoid(src_o).view(batch_size, -1)
        tgt_o = torch.sigmoid(tgt_o).view(batch_size, -1)

        # self-attention
        src_feats_pos = get_anchor_corrs(
            src, src_feats, self.config.km_clusters, dst='eu', iters=10, is_fast=True)[1]
        tgt_feats_pos = get_anchor_corrs(
            tgt, tgt_feats, self.config.km_clusters, dst='eu', iters=10, is_fast=True)[1]
        src_feats = self.sattn2(src_feats, src_feats_pos) + src_feats
        tgt_feats = self.sattn2(tgt_feats, tgt_feats_pos) + tgt_feats

        # clustering
        src_gamma, src_pi, src_node_xyz, src_node_feats = self.cluster(src, src_feats, src_o)
        tgt_gamma, tgt_pi, tgt_node_xyz, tgt_node_feats = self.cluster(tgt, tgt_feats, tgt_o)
        rot, trans, soft_corr_mu, soft_mu_scores = self.soft_svd(
            src_node_xyz, tgt_node_xyz, src_node_feats, tgt_node_feats, src_pi, tgt_pi)
        # iv_transf = soft_svd(tgt_mu_xyz, src_mu_xyz, tgt_mu_feats, src_mu_feats)
        # clustering-based loss
        src_clu_loss = self.cluloss(src, src_node_xyz.transpose(-1, -2), src_feats, src_gamma)
        tgt_clu_loss = self.cluloss(tgt, tgt_node_xyz.transpose(-1, -2), tgt_feats, tgt_gamma)
        # src_clu_loss = self.cluloss(src.transpose(-1, -2), src_log_scores.transpose(-1, -2), None)
        # tgt_clu_loss = self.cluloss(tgt.transpose(-1, -2), tgt_log_scores.transpose(-1, -2), None)
        clu_loss = 0.5 * (src_clu_loss + tgt_clu_loss)
        # we_loss = self.we_loss(src.transpose(-1, -2), tgt.transpose(-1, -2))
        # + self.we_loss(
        #     src_mu, soft_src, src_scores.transpose(-1, -2))
        loss = clu_loss
        if is_test:
            trans_init = integrate_trans(rot, trans)
            rot, trans = reg_solver(src, tgt, voxel_size=self.config.overlap_radius, trans_init=trans_init)

        return rot, trans, src_o, tgt_o, loss
