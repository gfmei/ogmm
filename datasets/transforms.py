#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2022 10:42 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : transforms.py
# @Software: PyCharm
import math
from typing import List, Dict

import torch
import torch.utils.data
import numpy as np

from scipy.spatial.distance import minkowski
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from sklearn.neighbors import NearestNeighbors

from datasets.datautils import uniform2sphere, se3_transform, so3_transform, se3_inv

""" gives some transform methods for 3d points """


def farthest_subsample_points(src, tgt, num_subsampled_points=768):
    src_xyz = src[:, :3]
    tgt_xyz = tgt[:, :3]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(src_xyz)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(tgt_xyz)
    random_p2 = random_p1
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return src[idx1, :], tgt[idx2, :]


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)


class OnUnitSphere:
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean

    def __call__(self, tensor):
        if self.zero_mean:
            m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
            v = tensor - m
        else:
            v = tensor
        nn = v.norm(p=2, dim=1)  # [N, D] -> [N]
        nmax = torch.max(nn)
        return v / nmax


class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0]  # [N, D] -> [D]
        s = torch.max(c)  # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        # return self.method1(tensor)
        return self.method2(tensor)


class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_matrix(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_matrix()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3_transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3_inv(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], _, _ = self.transform(sample['points'])
        else:
            src_transformed, transform_r_s, transform_s_r = self.transform(sample['points_src'])
            sample['transform_gt'] = transform_r_s  # Apply to source to get reference
            sample['points_src'] = src_transformed

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations
    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations
    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomTranslate(object):
    def __init__(self, mag=None, randomly=True):
        self.mag = 1.0 if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        t = torch.randn(1, 3).to(tensor)
        t = t / t.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = torch.eye(4).to(tensor)
        g[0:3, 3] = t[0, :]
        self.igt = g  # [4, 4]

        p1 = tensor + t
        return p1


def rand_rot_transl(rot_factor):
    anglex = np.random.uniform() * np.pi / rot_factor
    angley = np.random.uniform() * np.pi / rot_factor
    anglez = np.random.uniform() * np.pi / rot_factor
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
    rot = Rx.dot(Ry).dot(Rz)
    transl = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                       np.random.uniform(-0.5, 0.5)])
    euler = np.asarray([anglez, angley, anglex])

    return rot, transl, euler


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


def translate_pc(pts):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    trans_pts = np.add(np.multiply(pts, xyz1), xyz2).astype('float32')
    return trans_pts


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


class Jitter(object):
    """Jitter the position by a small amount

    Args:
        scale: Controls the amount to jitter. Noise will be sampled from
           a gaussian distribution with standard deviation given by scale,
           independently for each axis
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):
        for cloud in ['src_xyz', 'tgt_xyz']:
            noise = torch.randn(data[cloud].shape) * self.scale
            data[cloud] = data[cloud] + noise
        return data


class Resampler(object):
    def __init__(self, num: int = 1024):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
        elif points.shape[0] == k:
            rand_idxs = np.arange(points.shape[0])
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])

        return points[rand_idxs, :], rand_idxs

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'], rand_idx = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = 717  # This is a bug and should be removed, but is kept here to be consistent with Predator
                ref_size = 717
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            points_src, src_rand_idx = self._resample(sample['points_src'], src_size)
            points_ref, ref_rand_idx = self._resample(sample['points_ref'], ref_size)
            src_idx_map = np.full(sample['points_src'].shape[0], -1)
            ref_idx_map = np.full(sample['points_ref'].shape[0], -1)
            src_idx_map[src_rand_idx] = np.arange(src_size)
            ref_idx_map[ref_rand_idx] = np.arange(ref_size)

            correspondences = np.stack([src_idx_map[sample['correspondences'][0]],
                                        ref_idx_map[sample['correspondences'][1]]])
            correspondences = correspondences[:, np.all(correspondences >= 0, axis=0)]

            sample['correspondences'] = correspondences
            sample['points_src'] = points_src
            sample['points_ref'] = points_ref
            sample['src_overlap'] = sample['src_overlap'][
                src_rand_idx]  # Assume overlap stays the same after resampling
            sample['ref_overlap'] = sample['ref_overlap'][ref_rand_idx]

        return sample


class FixedResampler(Resampler):
    """Fixed resampling to always choose the first N points.
    Always deterministic regardless of whether the deterministic flag has been set
    """

    @staticmethod
    def _resample(points, k):
        multiple = k // points.shape[0]
        remainder = k % points.shape[0]

        resampled = np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)
        return resampled


class SplitSourceRef:
    """Clones the point cloud into separate source and reference point clouds"""

    def __call__(self, sample: Dict):
        sample['points_raw'] = sample.pop('points')
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points_src'] = sample['points_raw'].detach()
            sample['points_ref'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points_src'] = sample['points_raw'].copy()
            sample['points_ref'] = sample['points_raw'].copy()

        n_points = sample['points_raw'].shape[0]
        sample['correspondences'] = np.tile(np.arange(n_points), (2, 1))

        return sample


class RandomJitter:
    """ generate perturbations """

    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        if 'points' in sample:
            sample['points'] = self.jitter(sample['points'])
        else:
            sample['points_src'] = self.jitter(sample['points_src'])
            sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class RandomCrop:
    """
    Randomly crops the *source* point cloud.
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

        return points[mask, :], mask

    def __call__(self, sample):

        sample['crop_proportion'] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample  # No need crop

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if len(self.p_keep) == 1:
            points_src, src_mask = self.crop(sample['points_src'], self.p_keep[0])
            points_ref = sample['points_ref']
            ref_mask = np.ones(sample['points_ref'].shape[0], dtype=np.bool)
        else:
            points_src, src_mask = self.crop(sample['points_src'], self.p_keep[0])
            points_ref, ref_mask = self.crop(sample['points_ref'], self.p_keep[0])

        # Compute overlap masks
        src_overlap = np.zeros(sample['points_src'].shape[0], dtype=np.bool)
        temp = ref_mask[sample['correspondences'][1]]
        src_overlap[sample['correspondences'][0][temp]] = 1
        src_overlap = src_overlap[src_mask]

        ref_overlap = np.zeros(sample['points_ref'].shape[0], dtype=np.bool)
        temp = src_mask[sample['correspondences'][0]]
        ref_overlap[sample['correspondences'][1][temp]] = 1
        ref_overlap = ref_overlap[ref_mask]

        # Update correspondences
        src_idx_map = np.full(sample['points_src'].shape[0], -1)
        src_idx_map[src_mask] = np.arange(src_mask.sum())  # indicates index of new point for each original point index
        ref_idx_map = np.full(sample['points_ref'].shape[0], -1)
        ref_idx_map[ref_mask] = np.arange(ref_mask.sum())

        correspondences = np.stack([src_idx_map[sample['correspondences'][0]],
                                    ref_idx_map[sample['correspondences'][1]]])
        correspondences = correspondences[:, np.all(correspondences >= 0, axis=0)]

        sample['points_src'] = points_src
        sample['points_ref'] = points_ref
        sample['correspondences'] = correspondences
        sample['src_overlap'] = src_overlap
        sample['ref_overlap'] = ref_overlap

        return sample


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):
        if 'points' in sample:
            sample['points'] = np.random.permutation(sample['points'])
        else:
            ref_permute = np.random.permutation(sample['points_ref'].shape[0])
            src_permute = np.random.permutation(sample['points_src'].shape[0])

            sample['points_ref'] = sample['points_ref'][ref_permute, :]
            sample['points_src'] = sample['points_src'][src_permute, :]
            try:
                sample['ref_overlap'] = sample['ref_overlap'][ref_permute]
                sample['src_overlap'] = sample['src_overlap'][src_permute]
            except Exception:
                sample['ref_overlap'] = np.array(0.0)
                sample['src_overlap'] = np.array(0.0)

            ref_idx_map = np.full(sample['points_ref'].shape[0], -1)
            ref_idx_map[ref_permute] = np.arange(sample['points_ref'].shape[0])
            src_idx_map = np.full(sample['points_src'].shape[0], -1)
            src_idx_map[src_permute] = np.arange(sample['points_src'].shape[0])
            sample['correspondences'] = np.stack([
                src_idx_map[sample['correspondences'][0]],
                ref_idx_map[sample['correspondences'][1]]])

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True
        return sample
