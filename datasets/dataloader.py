#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/14/2022 11:38 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : dataloader.py
# @Software: PyCharm
import torchvision
from torch.utils.data import DataLoader

from datasets.modelnet import ModelNetCP, ModelNetDV, get_transforms, ModelNetHdf
from datasets.realdata import Scene7, IclNuim


def get_categories(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo


def data_loader(args):
    if args.dataset == '7scene':
        train_data = Scene7(args, root=args.root, partition='train')
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_data = Scene7(args, root=args.root, partition='test')
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=6)
    elif args.dataset == 'modelnetdv':
        train_loader = DataLoader(ModelNetDV(
            root=args.root, num_points=2048, num_subsampled_points=args.n_subsampled_points,
            partition='train', gaussian_noise=args.gaussian_noise, unseen=args.unseen, rot_factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_loader = DataLoader(ModelNetDV(
            root=args.root, num_points=2048, num_subsampled_points=args.n_subsampled_points,
            partition='test', gaussian_noise=args.gaussian_noise, unseen=args.unseen, rot_factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    elif args.dataset == 'icl_nuim':
        train_data = IclNuim(args, args.root, partition='train')
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_data = IclNuim(args, args.root, partition='test')
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    elif args.dataset == 'modelnetcp':
        train_data = ModelNetCP(args, args.root, partition='train')
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_data = ModelNetCP(args, args.root, partition='test')
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    elif args.dataset == 'modelnet':
        train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                                          args.n_points, args.partial)
        train_transforms = torchvision.transforms.Compose(train_transforms)
        val_transforms = torchvision.transforms.Compose(val_transforms)
        train_data = ModelNetHdf(args, args.root, partition='train', unseen=args.unseen, transform=train_transforms)
        val_data = ModelNetHdf(args, args.root, partition='test', unseen=args.unseen, transform=val_transforms)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_loader = DataLoader(val_data, args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    else:
        raise Exception("not implemented")
    return train_loader, test_loader
