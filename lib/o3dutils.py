#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2022 10:37 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : o3dutils.py
# @Software: PyCharm
import copy

import numpy as np
import open3d as o3d
import torch


def make_point_cloud(pts, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(pts))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(to_array(normals))
    return pcd


def make_color_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.ndim == 2
    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return pcd


def process_point_cloud(pcd, voxel_size, normals=None, ds=False):
    if normals is None:
        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if ds:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def pointcloud_to_spheres(pcd, voxel_size, color, sphere_size=0.6):
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd = np.array(pcd.points)
    for i, p in enumerate(pcd):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        # si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def get_matching_indices(source, target, trans, search_voxel_size, k=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if k is not None:
            idx = idx[:k]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                      voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - gamma: [4, 4] or [bs, 4, 4], SE3 transformSE3 matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def make_open3d_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def refine_registration(source, target, transformation, voxel_size, pl=False):
    source = make_open3d_point_cloud(source)
    target = make_open3d_point_cloud(target)
    distance_threshold = voxel_size * 2.0
    if pl:
        radius_normal = voxel_size * 2
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result.transformation


def reg_solver(src, tgt, voxel_size=0.05, trans_init=None):
    """
    Compute rigid transforms between two point sets with RANSAC
    Args: src_xyz (torch.Tensor): (B, 3, M) feats && tgt_xyz (torch.Tensor): (B, 3, N) feats
    Returns: Transform R (B, 3, 3), t (B, 3) to get from src_o to tgt_o, i.e. T*src_xyz = tgt_xyz
    """
    bs = src.shape[0]
    rot, transl = [], []
    src, tgt = src.transpose(-1, -2), tgt.transpose(-1, -2)
    for i in range(len(src)):
        src_i = src[i].detach().cpu().numpy()
        tgt_i = tgt[i].detach().cpu().numpy()
        trans_init_i = np.eye(4)
        if trans_init is not None:
            trans_init_i = trans_init[i].detach().cpu().numpy()
            # source, target, voxel_size=0.15, dis_thresh=0.01, model='icp', trans_init=None
        transformation = refine_registration(src_i, tgt_i, trans_init_i, voxel_size)
        rot.append(transformation[:3, :3].copy())
        transl.append(transformation[:3, -1].copy())
    rot = torch.cat([torch.from_numpy(r).unsqueeze(0) for r in rot], dim=0).float().cuda()
    transl = torch.cat([torch.from_numpy(t).reshape(1, 1, 3) for t in transl], dim=0).float().cuda()

    return rot, transl.view(bs, 3)


def overlap_labels(src, tgt, transform, thresh=0.05):
    correspondence = get_correspondences(
        make_point_cloud(src), make_point_cloud(tgt), transform, thresh)
    src_idx = list(set(correspondence[:, 0].int().tolist()))
    tgt_idx = list(set(correspondence[:, 1].int().tolist()))
    src_label = np.zeros(src.shape[0])
    src_label[src_idx] = 1.
    tgt_label = np.zeros(tgt.shape[0])
    tgt_label[tgt_idx] = 1.
    return src_label, tgt_label
