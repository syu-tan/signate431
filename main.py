#!/usr/bin/env python 3
# coding: utf-8

import sys
import random

import numpy as np
import torch
import deepspeed
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold

from lemon_utils.arg import manage_log, parse_arguments
from lemon_utils.train import train_model

def prepare(args):
    """ initialize model """

    # モデル作成
    from lemon_utils.model import make_model
    net = make_model(args)
    
    # 損失関数作成
    from lemon_utils.loss import make_criterion
    criterion = make_criterion(args)
    
    # 最適化法設定
    from lemon_utils.optim import make_optim
    net, optimizer, scheduler = make_optim(net=net, args=args)

    return net, criterion, optimizer, scheduler

def main(args):
    # 学習毎にログを管理する
    manage_log(args)


    # 乱数のシードを設定
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    


    # データセット作成
    from lemon_utils.data import make_datapath_list, LemonDataset, ImageTransform

    if args.disable_val:
        train_data, test_data = make_datapath_list(root_path=args.root_path, test_size=args.val_size, random_seed=args.random_seed, disable_val=args.disable_val, use_denoise=args.use_denoise, denoise_path=args.denoise_path)
        
        train_dataset = LemonDataset(file_list=train_data, transform=ImageTransform(args), phase='train')
        test_dataset = LemonDataset(file_list=test_data, transform=ImageTransform(args), phase='test')
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
        
        dataloaders_dict = {"train": train_dataloader, "val": None, "test": test_dataloader}

        if args.phase == 'train':
            net, criterion, optimizer, scheduler = prepare(args)
            # 学習・検証を実行する
            net = train_model(net, dataloaders_dict, criterion, optimizer, scheduler, -1, args=args)
        
        
    else:
        if args.is_k_fold:
            # k-fold で検証
            train_data, tets_data = make_datapath_list(root_path=args.root_path, test_size=args.val_size, random_seed=args.random_seed, disable_val=args.is_k_fold, use_denoise=args.use_denoise, denoise_path=args.denoise_path) # 全て train data に回して k-fold

            # DataSetを作成
            train_dataset = LemonDataset(file_list=train_data, transform=ImageTransform(args), phase='train')
            test_dataset = LemonDataset(file_list=test_data, transform=ImageTransform(args), phase='test')

            kf = KFold(n_splits=args.k_fold_num, shuffle=False)

            net_list = []

            for k_fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):

                print(f'>>> k-fold {k_fold}/{args.k_fold_num} ')
                k_train_dataset = Subset(train_dataset, train_index)
                k_val_dataset   = Subset(train_dataset, val_index)
                print(f'>>> k-fold train : {len(k_train_dataset)} val : {len(k_val_dataset)}')

                # DataLoaderを作成
                k_train_dataloader = torch.utils.data.DataLoader(
                    k_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)

                k_val_dataloader = torch.utils.data.DataLoader(
                    k_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
                

                # 辞書型変数にまとめる
                dataloaders_dict = {"train": k_train_dataloader, "val": k_val_dataloader}

                if args.phase == 'train':
                    net, criterion, optimizer, scheduler = prepare(args)
                    # 学習・検証を実行する
                    net = train_model(net, dataloaders_dict, criterion, optimizer, scheduler, k_fold, args=args)
                    net_list.append(net)

            # k-fold で推論
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
            dataloaders_dict = {"test": test_dataloader}

            from lemon_utils.test import test_model
            test_model(net_list, dataloaders_dict, args=args)



        # 普通に train valid splite で検証

        train_data, val_data, test_data = make_datapath_list(root_path=args.root_path, test_size=args.val_size, random_seed=args.random_seed, use_denoise=args.use_denoise, denoise_path=args.denoise_path)

        # DataSetを作成
        train_dataset = LemonDataset(file_list=train_data, transform=ImageTransform(args), phase='train')
        val_dataset = LemonDataset(file_list=val_data, transform=ImageTransform(args), phase='val')
        test_dataset = LemonDataset(file_list=test_data, transform=ImageTransform(args), phase='test')

        # DataLoaderを作成
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
    

        # 辞書型変数にまとめる
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

        if args.phase == 'train':
            net, criterion, optimizer, scheduler = prepare(args)
            # 学習・検証を実行する
            net = train_model(net, dataloaders_dict, criterion, optimizer, scheduler, -1, args=args)



    if args.phase == "test":
        # 推論だけの場合
        net, _,_,_ = prepare(args)
        
        if args.is_ensemble:
            # アンサンブルのとき
            from lemon_utils.model import EnsembleModel
            net = EnsembleModel(args)

        else:
            # 単一モデルの場合
            weights = torch.load(args.test_model_path)
            net.load_state_dict(weights)
    
    # 学習したモデルで推論
    from lemon_utils.test import test_model
    test_model(net, dataloaders_dict, args=args)


if __name__ == "__main__":
    """
    NOTE(train example)
    $ python main.py
    """

    main(parse_arguments(sys.argv[1:]))