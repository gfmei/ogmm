#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/23/2021 3:30 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : datautils.py
# @Software: PyCharm
import glob
import os
import re
from os import listdir
from os.path import join, isfile, splitext, isdir
from typing import List

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in tgt_xyz for each point in src_xyz
    Input:
        src_xyz: Nxm array of points
        tgt_xyz: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: tgt_xyz indices of the nearest neighbor
    '''

    assert src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def get_inner_labels(src, tgt, transf, thresh=0.05, label_type=3):
    perm_mat = np.zeros((src.shape[0], tgt.shape[0]))
    inlier_src = np.zeros((src.shape[0], 1))
    inlier_tgt = np.zeros((tgt.shape[0], 1))
    points_src_transform = transform(transf, src[:, :3])
    points_tgt = tgt[:, :3]
    dist_s2et, indx_s2et = nearest_neighbor(points_src_transform, points_tgt)
    dist_t2es, indx_t2es = nearest_neighbor(points_tgt, points_src_transform)
    if label_type == 1:
        for row_i in range(src.shape[0]):
            if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < thresh:
                perm_mat[row_i, indx_s2et[row_i]] = 1
    elif label_type == 2:
        for row_i in range(src.shape[0]):
            if dist_s2et[row_i] < thresh:
                perm_mat[row_i, indx_s2et[row_i]] = 1
        for col_i in range(tgt.shape[0]):
            if dist_t2es[col_i] < thresh:
                perm_mat[indx_t2es[col_i], col_i] = 1
    elif label_type == 3:  # 双边对应填充, 完全填充, 双边对应填充+部分对应填充
        for row_i in range(src.shape[0]):
            if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < thresh:
                perm_mat[row_i, indx_s2et[row_i]] = 1
        for row_i in range(src.shape[0]):
            if (np.sum(perm_mat[row_i, :]) == 0 and np.sum(perm_mat[:, indx_s2et[row_i]]) == 0
                    and dist_s2et[row_i] < thresh):
                perm_mat[row_i, indx_s2et[row_i]] = 1
        for col_i in range(tgt.shape[0]):
            # print(col_i, tgt_xyz.shape[0])
            if (np.sum(perm_mat[:, col_i]) == 0
                    and np.sum(perm_mat[indx_t2es[col_i], :]) == 0
                    and dist_t2es[col_i] < thresh):
                perm_mat[indx_t2es[col_i], col_i] = 1
        outlier_src_ind = np.where(np.sum(perm_mat, axis=1) == 0)[0]
        outlier_tgt_ind = np.where(np.sum(perm_mat, axis=0) == 0)[0]
        points_src_transform_rest = points_src_transform[outlier_src_ind]
        points_tgt_rest = points_tgt[outlier_tgt_ind]
        if points_src_transform_rest.shape[0] > 0 and points_tgt_rest.shape[0] > 0:
            dist_s2et, indx_s2et = nearest_neighbor(points_src_transform_rest, points_tgt_rest)
            dist_t2es, indx_t2es = nearest_neighbor(points_tgt_rest, points_src_transform_rest)
            for row_i in range(points_src_transform_rest.shape[0]):
                if indx_t2es[indx_s2et[row_i]] == row_i and dist_s2et[row_i] < thresh * 2:
                    perm_mat[outlier_src_ind[row_i], outlier_tgt_ind[indx_s2et[row_i]]] = 1
    inlier_src_ind = np.where(np.sum(perm_mat, axis=1))[0]
    inlier_ref_ind = np.where(np.sum(perm_mat, axis=0))[0]
    inlier_src[inlier_src_ind] = 1
    inlier_tgt[inlier_ref_ind] = 1

    return perm_mat, inlier_src, inlier_tgt


def uniform2sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def write_trajectory(traj, metadata, filename, dim=4):
    """
    Writes the trajectory into a '.txt' file in 3DMatch/Redwood format.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    traj (numpy array): trajectory for n pairs[n,dim, dim]
    metadata (numpy array): file containing metadata about fragment numbers [n,3]
    filename (str): path where to save the '.txt' file containing trajectory data
    dim (int): dimension of the transformSE3 matrix (4x4 for 3D data)
    """

    with open(filename, 'w') as f:
        for idx in range(traj.shape[0]):
            # Only save the transfromation parameters for which the overlap threshold was satisfied
            if metadata[idx][2]:
                p = traj[idx, :, :].tolist()
                f.write('\t'.join(map(str, metadata[idx])) + '\n')
                f.write('\n'.join('\t'.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
                f.write('\n')


def load_data(partition, root):
    data_dir = os.path.join(root, '')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def knn_idx(pts, k):
    kdt = cKDTree(pts)
    _, idx = kdt.query(pts, k=k + 1)
    return idx[:, 1:]


def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2 * np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)


def so3_transform(rot, xyz):
    """

    Args:
        rot: ([B,] 3, 3)
        xyz: ([B,] N, 3)

    Returns:

    """
    assert xyz.shape[-1] == 3 and rot.shape[:-2] == xyz.shape[:-2]
    transformed = np.einsum('...ij,...bj->...bi', rot, xyz)
    return transformed


def se3_transform(pose, xyz):
    """Apply rigid transformation to points

    Args:
        pose: ([B,] 3, 4)
        xyz: ([B,] N, 3)

    Returns:

    """

    assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

    return transformed


def se3_init(rot, trans):
    pose = np.concatenate([rot, trans], axis=-1)
    return pose


def se3_inv(pose):
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords, feats):
        for transform in self.transforms:
            coords, feats = transform(coords, feats)
        return coords, feats


def farthest_point_sample(point, npoint, is_idx=False):
    """
    Input:
        src_xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    if is_idx:
        return point, centroids.astype(np.int32)
    return point


class RandomCrop(object):
    """Randomly crops the *tgt_xyz* point cloud, approximately retaining half the points
    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """

    def __init__(self, p_keep: List = None):
        if p_keep is None:
            p_keep = [0.7, 0.7]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform2sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, src, tgt, seed=None):
        if np.all(self.p_keep == 1.0):
            return src, tgt  # No need crop

        if seed is not None:
            np.random.seed(seed)

        if len(self.p_keep) == 1:
            src = self.crop(src, self.p_keep[0])
        else:
            src = self.crop(src, self.p_keep[0])
            tgt = self.crop(tgt, self.p_keep[1])
        return src, tgt


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = score_mat.cpu().numpy()
    if score_mat.ndim == 2:
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o755)


def sorted_alphanum(file_list_ordered):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            join(path, f) for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def get_file_list_specific(path, color_depth, extension=None):
    if extension is None:
        file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            join(path, f) for f in listdir(path)
            if isfile(join(path, f)) and color_depth in f and splitext(f)[1] == extension
        ]
        file_list = sorted_alphanum(file_list)
    return file_list


def get_folder_list(path):
    folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    folder_list = sorted_alphanum(folder_list)
    return folder_list


def read_trajectory(filename, dim=4):
    class CameraPose:
        def __init__(self, meta, mat):
            self.metadata = meta
            self.pose = mat

        def __str__(self):
            return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
                   "pose : " + "\n" + np.array_str(self.pose)

    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(dim, dim))
            for i in range(dim):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
        return traj


def identity():
    return np.eye(3, 4)


def transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformSE3 matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed


def inverse(rot: np.ndarray, trans: np.ndarray):
    """Returns the inverse of the SE3 transform

    Args:
        rot: ([B,] 3, 3) transform

    Returns:
        ([B,] 3/4, 4) matrix containing the inverse

    """
    # rot_pred = g[..., :3, :3]  # (3, 3)
    # trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    # if g.shape[-2] == 4:
    #     inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def from_xyzquat(xyzquat):
    """Constructs SE3 matrix from src_xyz, y, z, qx, qy, qz, qw

    Args:
        xyzquat: np.array (7,) containing translation and quaterion

    Returns:
        SE3 matrix (4, 4)
    """
    rot = Rotation.from_quat(xyzquat[3:])
    trans = rot.apply(-xyzquat[:3])
    transform = np.concatenate([rot.as_dcm(), trans[:, None]], axis=1)
    transform = np.concatenate([transform, [[0.0, 0.0, 0.0, 1.0]]], axis=0)

    return transform


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array
