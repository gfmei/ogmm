#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/12/2022 10:29 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : modelnet.py
# @Software: PyCharm
import os
from typing import List

import numpy as np
import torch

import datasets.transforms as Transforms
from torch.utils.data import Dataset

from datasets.datautils import RandomCrop, farthest_point_sample, load_data, get_rri
from lib.o3dutils import overlap_labels
from datasets.transforms import jitter_pcd, random_pose
from lib.se3 import np_mat2quat


def get_transforms(noise_type: str, rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.Resampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.Resampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % zipfile)


class ModelNetGMR(Dataset):
    def __init__(self, root, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNetGMR, self).__init__()
        self.data, self.label = load_data(partition, root)
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        self.crop = RandomCrop(p_keep=[0.55, 0.55])
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen and self.partition == 'test':
            self.data = self.data[self.label >= 20]
            self.label = self.label[self.label >= 20]
        else:
            self.data = self.data[self.label < 20]
            self.label = self.label[self.label < 20]
        self.n_points = num_points
        self.max_angle = np.pi / rot_factor
        self.max_trans = 0.5
        self.noisy = gaussian_noise
        self.k = 20
        self.get_rri = get_rri

    def __getitem__(self, index):
        if self.partition != 'train':
            np.random.seed(index)
        points = self.data[index]
        src = np.random.permutation(points[:, :3])[:self.n_points]
        tgt = np.random.permutation(points[:, :3])[:self.n_points]

        if self.subsampled:
            src, tgt = self.crop(src, tgt)
            if self.num_subsampled_points < src.shape[0]:
                src = farthest_point_sample(src, self.num_subsampled_points, is_idx=False)
                tgt = farthest_point_sample(tgt, self.num_subsampled_points, is_idx=False)

        transform = random_pose(self.max_angle, self.max_trans / 2)
        if self.partition == 'train':
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
        else:
            tgt = tgt @ transform[:3, :3].T + transform[:3, 3]
        if self.noisy:
            src = jitter_pcd(src)
            tgt = jitter_pcd(tgt)
        src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
        tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)
        return src.astype('float32'), tgt.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class ModelNetCP(Dataset):
    def __init__(self, args, root, partition='train', category=None):
        super(ModelNetCP, self).__init__()
        self.data, self.label = load_data(partition, root)
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]
        self.num_points = args.n_points
        self.partition = partition
        self.max_angle = args.rot_mag
        self.max_trans = args.trans_mag
        self.unseen = args.unseen
        self.label = self.label.squeeze()
        self.crop = RandomCrop(p_keep=[args.partial[0]])
        if self.unseen and self.partition == 'test':
            self.data = self.data[self.label >= 20]
            self.label = self.label[self.label >= 20]
        else:
            self.data = self.data[self.label < 20]
            self.label = self.label[self.label < 20]
        self.noisy = args.noise_type

    def __getitem__(self, item):
        points = self.data[item]
        src = np.random.permutation(points[:, :3])[:self.num_points]
        tgt = np.random.permutation(points[:, :3])[:self.num_points]

        if self.partition != 'train':
            np.random.seed(item)
        src, tgt = self.crop(src, tgt)
        transform = random_pose(self.max_angle, self.max_trans / 2)
        if self.partition == 'train':
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
        else:
            tgt = tgt @ transform[:3, :3].T + transform[:3, 3]
        if self.noisy != 'clean':
            src = jitter_pcd(src)
            tgt = jitter_pcd(tgt)
        src_label, tgt_label = overlap_labels(src, tgt, transform)

        sample_out = {
            'src_xyz': src[:, :3].astype('float32'),
            'tgt_xyz': tgt[:, :3].astype('float32'),
            'tgt_raw': points[:, :3].astype('float32'),
            'src_overlap': src_label.astype('float32'),
            'tgt_overlap': tgt_label.astype('float32'),
            # 'correspondences': torch.from_numpy(sample['correspondences']),
            'transform_gt': transform.astype('float32'),
            "pose_gt": np_mat2quat(transform.astype('float32').astype('float32')),
            'idx': np.array(item),
            # 'corr_xyz': torch.from_numpy(corr_xyz),
        }

        return sample_out

    def __len__(self):
        return self.data.shape[0]


class ModelNetDV(Dataset):
    def __init__(self, root, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None):
        super(ModelNetDV, self).__init__()
        self.data, self.label = load_data(partition, root)
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.max_angle = np.pi / rot_factor
        self.max_trans = 0.5
        self.noisy = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        self.crop = RandomCrop(p_keep=[0.75, 0.75])
        if self.unseen and self.partition == 'test':
            self.data = self.data[self.label >= 20]
            self.label = self.label[self.label >= 20]
        else:
            self.data = self.data[self.label < 20]
            self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        points = self.data[item][:self.num_points]
        src = np.random.permutation(points[:, :3])[:self.num_points]
        tgt = np.random.permutation(points[:, :3])[:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)

        if self.subsampled:
            src, tgt = self.crop(src, tgt)
            if self.num_subsampled_points < src.shape[0]:
                src = farthest_point_sample(src, self.num_subsampled_points, is_idx=False)
            if self.num_subsampled_points < tgt.shape[0] // 2:
                tgt = farthest_point_sample(tgt, 2 * self.num_subsampled_points, is_idx=False)

        transform = random_pose(self.max_angle, self.max_trans / 2)
        if self.partition == 'train':
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
        else:
            tgt = tgt @ transform[:3, :3].T + transform[:3, 3]
        if self.noisy != 'clean':
            src = jitter_pcd(src)
            tgt = jitter_pcd(tgt)

        return [src.astype('float32'), tgt.astype('float32'), transform.astype('float32')]

    def __len__(self):
        return self.data.shape[0]


class ModelNetHdf(Dataset):
    def __init__(self, args, root, unseen=False, transform=None, partition='train', category=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available
        Args:
            root (str): Folder containing processed dataset
            partition (str): Dataset subset, either 'train' or 'test'
            category (list): Categories to use
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.config = args
        self.root = root
        self.overlap_radius = args.overlap_radius

        self.data, self.label = load_data(partition, root)
        self.label = self.label.squeeze()
        if category is not None:
            self.data = self.data[self.label == category]
            self.label = self.label[self.label == category]

        self.unseen = unseen
        self.data, self.labels = load_data(partition, root)
        # if self.unseen and partition == 'test':
        #     self.data = self.data[self.label >= 20]
        #     self.label = self.label[self.label >= 20]
        # else:
        #     self.data = self.data[self.label < 20]
        #     self.label = self.label[self.label < 20]
        self._transform = transform

    def __getitem__(self, item):
        points = np.random.permutation(self.data[item, :, :])[:1024]
        sample = {'points': points, 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        # Apply perturbation
        if self._transform:
            sample = self._transform(sample)

        # corr_xyz = np.concatenate([
        #     sample['points_src'][sample['correspondences'][0], :3],
        #     sample['points_ref'][sample['correspondences'][1], :3]], axis=1)

        sample_out = {
            'src_xyz': torch.from_numpy(sample['points_src'][:, :3].astype('float32')),
            'tgt_xyz': torch.from_numpy(sample['points_ref'][:, :3].astype('float32')),
            'tgt_raw': torch.from_numpy(sample['points_raw'][:, :3].astype('float32')),
            'src_overlap': torch.from_numpy(sample['src_overlap'].astype('float32')),
            'tgt_overlap': torch.from_numpy(sample['ref_overlap'].astype('float32')),
            # 'correspondences': torch.from_numpy(sample['correspondences']),
            'transform_gt': torch.from_numpy(sample['transform_gt'].astype('float32')),
            "pose_gt": torch.from_numpy(np_mat2quat(sample['transform_gt'].astype('float32'))),
            'idx': torch.from_numpy(sample['idx']),
            # 'corr_xyz': torch.from_numpy(corr_xyz),
        }

        return sample_out

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print('hello world')
