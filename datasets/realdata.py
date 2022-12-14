#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/14/2022 11:39 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : realdata.py
# @Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import glob
import os

import h5py
import numpy as np
import six
from torch.utils.data.dataset import Dataset

from datasets import mesh
from datasets.datautils import RandomCrop, farthest_point_sample
from datasets.transforms import random_pose
from lib.o3dutils import overlap_labels
from lib.se3 import np_mat2quat


def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []
    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


class Scene7(Dataset):
    """ [Scene7 PointCloud](https://github.com/XiaoshuiHuang/fmr) """

    def __init__(self, args, root, partition='test', train=1):
        super().__init__()
        if train > 0:
            pattern = '*.ply'
        elif train == 0:
            pattern = '*.ply'
        else:
            pattern = ['*.ply', '*.ply']
        if partition == "test":
            classes = ["7-scenes-office"]
        else:
            classes = ["7-scenes-chess", "7-scenes-fire", "7-scenes-heads", "7-scenes-pumpkin",
                       "7-scenes-redkitchen", "7-scenes-stairs"]
        rootdir = os.path.join(root, '7scene')
        if isinstance(pattern, six.string_types):
            pattern = [pattern]
        # find all the class names
        self.args = args
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # get all the 3D point cloud paths for the class of class_to_idx
        self.samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not self.samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))
        self.classes = classes
        self.partition = partition
        self.rot_factor = 4.0
        self.max_trans = 0.5
        self.crop = RandomCrop(p_keep=args.partial)
        self.n_points = args.n_points
        self.loader = mesh.plyread

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        """
        path, _ = self.samples[index]
        points = np.array(self.loader(path).vertex_array)[:self.n_points].astype('float32')
        if points.shape[0] > self.n_points:
            src = np.random.permutation(points)[:self.n_points]
            tgt = np.random.permutation(points)[:self.n_points]  # [n, 3]
        else:
            src = copy.deepcopy(points)
            tgt = copy.deepcopy(points)
        transform = random_pose(np.pi / self.rot_factor, self.max_trans)
        tgt = tgt @ transform[:3, :3].T + transform[:3, 3]
        src, tgt = self.crop(src, tgt)
        num_subsampled_points = int(self.args.partial[0] * self.n_points)
        if src.shape[0] != num_subsampled_points:
            src = farthest_point_sample(src, num_subsampled_points, is_idx=False)
        if tgt.shape[0] != num_subsampled_points:
            tgt = farthest_point_sample(tgt, num_subsampled_points, is_idx=False)
        src_overlap, tgt_overlap = overlap_labels(src, tgt, transform)
        # Transform to my format
        sample_out = {
            'src_xyz': src.astype('float32'),
            'tgt_xyz': tgt.astype('float32'),
            'tgt_raw': points.astype('float32'),
            'src_overlap': src_overlap.astype('float32'),
            'tgt_overlap': src_overlap.astype('float32'),
            'transform_gt': transform.astype('float32'),
            "pose_gt": np_mat2quat(transform.astype('float32')),
            'idx': np.array(index).astype('float32'),
        }
        return sample_out

    def __len__(self):
        return len(self.samples)


class IclNuim(Dataset):
    def __init__(self, args, root, partition='test'):
        super(IclNuim, self).__init__()
        d_path = os.path.join(root, 'icl_nuim', partition, 'icl_nuim.h5')
        if partition == 'test':
            with h5py.File(d_path, 'r') as f:
                self.source = f['source'][...]
                self.target = f['target'][...]
                self.transform = f['transform'][...]
        else:
            with h5py.File(d_path, 'r') as f:
                self.source = f['points'][...]
                self.target = None
                self.transform = None
        self.max_angle = 45.0 / 180 * np.pi
        self.max_trans = 0.5
        self.crop = RandomCrop(p_keep=args.partial)
        self.n_points = args.n_points
        self.partition = partition
        self.args = args

    def __getitem__(self, index):
        np.random.seed(index)
        if self.partition == 'test':
            transform = self.transform[index]
            src = self.source[index][:self.n_points]
            src = src @ transform[:3, :3].T + transform[:3, 3]
            tgt = self.target[index][:self.n_points]
        else:
            src = self.source[index]
            tgt = copy.deepcopy(self.source[index])
            src = np.random.permutation(src)[:self.n_points]
            tgt = np.random.permutation(tgt)[:self.n_points]
        src, tgt = self.crop(src, tgt)
        num_subsampled_points = int(self.args.partial[0] * self.n_points)
        if src.shape[0] != num_subsampled_points:
            src = farthest_point_sample(src, num_subsampled_points, is_idx=False)
        if tgt.shape[0] != num_subsampled_points:
            tgt = farthest_point_sample(tgt, num_subsampled_points, is_idx=False)
        transform = random_pose(self.max_angle, self.max_trans)
        tgt = tgt @ transform[:3, :3].T + transform[:3, 3]
        src_overlap, tgt_overlap = overlap_labels(src, tgt, transform, thresh=0.075)

        # Transform to my format
        sample_out = {
            'src_xyz': src.astype('float32'),
            'tgt_xyz': tgt.astype('float32'),
            'tgt_raw': self.source[index].astype('float32'),
            'src_overlap': src_overlap.astype('float32'),
            'tgt_overlap': src_overlap.astype('float32'),
            'transform_gt': transform.astype('float32'),
            "pose_gt": np_mat2quat(transform.astype('float32')),
            'idx': np.array(index).astype('float32'),
        }

        return sample_out

    def __len__(self):
        return self.source.shape[0]
