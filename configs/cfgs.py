#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/14/2022 11:37 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : cfgs.py
# @Software: PyCharm

import argparse


def mnet():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    # dataset set
    parser.add_argument('--root', type=str, default='/data/gmei/data', metavar='N', help='path of data')
    parser.add_argument('--dataset', type=str, default='modelnet', choices=[
        'modelnet', 'modelnetdv', 'regtr', 'icl_nuim', 'modelnetcp'], metavar='N', help='dataset to use')
    parser.add_argument('--trans_mag', type=float, default=0.5, metavar='N', help='translation')
    parser.add_argument('--rot_mag', type=float, default=45, metavar='N', help='rotation')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N', help='Whether to test on unseen category')
    parser.add_argument('--n_points', type=int, default=717, metavar='N', help='Num of points to use')
    parser.add_argument('--partial', type=list, default=[0.70, 0.70], metavar='N', help='Whether to use tnet')
    parser.add_argument('--noise_type', type=str, default='crop', metavar='N', help='Whether to use tnet')
    parser.add_argument('--overlap_radius', type=float, default=0.0375, metavar='N', help='')

    # model set
    parser.add_argument('--model', type=str, default='GMMReg', metavar='N', help='Model to use')
    parser.add_argument('--attn', type=str, default='transformer', metavar='N', choices=['identity', 'transformer'],
                        help='Attention-based attn generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N', choices=['mlp', 'svd'],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=320, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--km_clusters', type=int, default=128, metavar='N', help='Number of clusters for kmeans')
    parser.add_argument('--n_clusters', type=int, default=32, metavar='N', help='Number of clusters')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--num_heads', type=int, default=4, metavar='N', help='Num of heads in multi_headed_attention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--K', type=int, default=20, metavar='N', help='Num of neighbors to use for DGCNN')
    parser.add_argument('--gnn_k', type=int, default=20, metavar='N', help='Num of neighbors to use for DGCNN')
    parser.add_argument('--tau', type=float, default=0.01, metavar='N', help='')
    parser.add_argument('--nn_margin', type=float, default=0.5, metavar='N', help='')
    parser.add_argument('--loss_margin', type=float, default=0.012, metavar='N', help='')
    parser.add_argument('--n_keypoints', type=int, default=256, metavar='N', help='')
    parser.add_argument('--list_k1', type=list, default=[5, 5, 5], metavar='N', help='')
    parser.add_argument('--list_k2', type=list, default=[5, 5, 5], metavar='N', help='')
    # training set
    parser.add_argument('--exp_name', type=str, default='GMMReg32', metavar='N', help='Name of the experiment')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--iter', type=int, default=4, metavar='iter', help='Number of iter)')
    parser.add_argument('--epochs', type=int, default=450, metavar='N', help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='number of start training')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')

    # eval setting
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_plot_freq', type=int, default=10)
    return parser.parse_args()


def indoor():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    # dataset set
    parser.add_argument('--root', type=str, default='/data/gmei/data', metavar='N', help='path of data')
    parser.add_argument('--dataset', type=str, default='7scene', choices=['7scene', 'icl_nuim'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--trans_mag', type=float, default=0.5, metavar='N', help='translation')
    parser.add_argument('--rot_mag', type=float, default=45, metavar='N', help='rotation')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N', help='Whether to test on unseen category')
    parser.add_argument('--n_points', type=int, default=50000, metavar='N', help='Num of points to use')
    parser.add_argument('--partial', type=list, default=[0.70, 0.70], metavar='N', help='Whether to use tnet')
    parser.add_argument('--noise_type', type=str, default='crop', metavar='N', help='Whether to use tnet')
    parser.add_argument('--overlap_radius', type=float, default=0.0375, metavar='N', help='')

    # model set
    parser.add_argument('--model', type=str, default='GMMReg', metavar='N', help='Model to use')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N', choices=['dgcnn'], help='Encoder')
    parser.add_argument('--attn', type=str, default='transformer', metavar='N', choices=['identity', 'transformer'],
                        help='Attention-based attn generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N', choices=['mlp', 'svd'],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--dim', type=int, default=16, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--n_clusters', type=int, default=16, metavar='N', help='Number of clusters')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N', help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N', help='Num of heads in multi_headed_attention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N', help='Num of dimensions of fc in transformer')
    parser.add_argument('--K', type=int, default=20, metavar='N', help='Num of neighbors to use for DGCNN')
    parser.add_argument('--tau', type=float, default=0.01, metavar='N', help='')
    parser.add_argument('--nn_margin', type=float, default=0.5, metavar='N', help='')
    parser.add_argument('--loss_margin', type=float, default=0.01, metavar='N', help='')
    parser.add_argument('--n_keypoints', type=int, default=256, metavar='N', help='')
    parser.add_argument('--list_k1', type=list, default=[32, 32, 32], metavar='N', help='')
    parser.add_argument('--list_k2', type=list, default=[10, 10, 10], metavar='N', help='')
    parser.add_argument('--reg', type=float, default=0.1, metavar='N', help='Dropout ratio in transformer')
    parser.add_argument('--use_tnet', type=bool, default=False, metavar='N', help='Whether to use tnet')
    parser.add_argument('--k', type=int, default=20)

    # training set
    parser.add_argument('--exp_name', type=str, default='GMMReg', metavar='N', help='Name of the experiment')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--iter', type=int, default=4, metavar='iter', help='Number of iter)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='number of start training')
    parser.add_argument('--use_sgd', action='store_true', default=False, help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluate the model')

    # eval setting
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_plot_freq', type=int, default=10)
    return parser.parse_args()
