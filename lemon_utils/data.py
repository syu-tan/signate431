#!/usr/bin/env python 3
# coding: utf-8

import os
import glob
import os.path as osp
import random
import numpy as np
import json
import csv
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as albu
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms




class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, args):
        
        # ライブラリーによるデータ増強
        # https://github.com/albumentations-team/albumentations
        train_albu_aug_list = []
        train_torch_aug_list = []
        test_albu_preprocess_list = []
        
        if args.aug_croplemon:
            train_albu_aug_list.append(CropLemon(p=1))
            train_albu_aug_list.append(albu.Resize(args.size, args.size))
            test_albu_preprocess_list.append(CropLemon(p=1))

        if args.aug_rotate:
            train_albu_aug_list.append(albu.Rotate(limit=args.aug_rorate_limit,p=args.aug_rorate_p))
        
        if args.aug_random_gamma:
            train_albu_aug_list.append(albu.RandomGamma(gamma_limit=args.aug_random_gamma_limit, p=args.aug_random_gamma_p))

        if args.aug_random_brightness:
            train_albu_aug_list.append(albu.RandomBrightnessContrast(p=args.aug_random_brightness_p))

        if args.aug_elastic:
            train_albu_aug_list.append(albu.ElasticTransform(alpha=1,sigma=50,alpha_affine=50,p=args.aug_elastic_p))

        if args.aug_blur:
            train_albu_aug_list.append(albu.Blur(blur_limit=7, always_apply=False, p=args.aug_blur_p))

        if args.aug_cutout:
            albu.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=args.aug_cotout_p)

        if args.aug_downscale:
            train_albu_aug_list.append(albu.Downscale(scale_min=args.aug_downscale_min, scale_max=args.aug_downscale_max, interpolation=0, always_apply=False, p=args.aug_downscale_p))

        if args.aug_fancy_pca:
            train_albu_aug_list.append(albu.FancyPCA(alpha=0.1, always_apply=False, p=args.aug_fancy_pca_p))

        if args.aug_gaussnoise:
            train_albu_aug_list.append(albu.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=args.aug_gaussnoise_p))

        if args.aug_random_contrast:
            train_albu_aug_list.append(albu.RandomContrast(limit=0.2, always_apply=False, p=0.5))

        if args.aug_random_shadow:
            train_albu_aug_list.append(albu.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=args.aug_random_shadow_p))

        if args.aug_random_resize_crop:
            train_torch_aug_list.append(transforms.RandomResizedCrop(args.size, scale=args.aug_random_resize_crop_scale, ratio=args.aug_random_resize_crop_ratio))

        if args.aug_horizontalflip:
            train_torch_aug_list.append(transforms.RandomHorizontalFlip(p=args.aug_horizontalflip_p))

        if args.aug_verticalflip:
            train_torch_aug_list.append(transforms.RandomVerticalFlip(p=args.aug_verticalflip_p))

        train_torch_aug_list.append(transforms.Lambda(self.albumentations_transform))
        train_torch_aug_list.append(transforms.ToTensor()) # テンソルに変換
        train_torch_aug_list.append(transforms.Normalize(args.mean, args.std)) # 標準化

        test_preprocess_list = []
        test_preprocess_list.append(transforms.Lambda(self.test_albumentations_transform))

        test_preprocess_list.append(transforms.Resize(args.size)) # リサイズ

        if args.test_preprocess_centercrop:
            test_preprocess_list.append(transforms.CenterCrop(args.size)) # 画像中央をresize×resizeで切り取り

        test_preprocess_list.append(transforms.ToTensor())
        test_preprocess_list.append(transforms.Normalize(args.mean, args.std))

        self.albu_transforms = albu.Compose(train_albu_aug_list)
        self.test_albu_transforms = albu.Compose(train_albu_aug_list)
        
        self.data_transform = {
            'train': transforms.Compose(train_torch_aug_list),
            'val': transforms.Compose(test_preprocess_list),
            'test': transforms.Compose(test_preprocess_list),
        }
        
    def albumentations_transform(self, image):
        # albumemtations の関数化のため
        image_np = np.array(image)
        augmented = self.albu_transforms(image=image_np)
        image = Image.fromarray(augmented['image'])
        return image
    
    def test_albumentations_transform(self, image):
        # albumemtations の関数化のため
        image_np = np.array(image)
        augmented = self.test_albu_transforms(image=image_np)
        image = Image.fromarray(augmented['image'])
        return image

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)


def make_datapath_list(root_path, test_size=0.1, random_seed=1234, disable_val=False, phase='train', use_denoise=False, denoise_path='/home/syu/src/signate/431/searnch/noise_data.csv'):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    root_path : 学習データのルートパス（431 の場所）

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    print(f"読み込みデータ ==> {root_path}")
    
    path_list = []  # ここに格納する

    if use_denoise:
        with open(denoise_path) as f:
            reader = csv.reader(f)
            # ヘッダーの削除処理
            noise = [row[0] for row in reader][1:]

    
    with open(root_path + '/train_images.csv') as f:
        reader = csv.reader(f)
        
        # ヘッダーの削除処理
        reader = [row for row in reader][1:]
        
        for row in reader:
            if use_denoise and row[0] in noise:
                # print(row[0])
                continue
            # データのパスとラベル情報をセットにして格納
            path_list.append((root_path + f"/train_images/{row[0]}",int(row[1])))
            
    test_path_list = []
            
    with open(root_path + '/test_images.csv') as f:
        reader = csv.reader(f)
        
        # ヘッダーの削除処理
        reader = [row for row in reader][1:]
        
        for row in reader:
            # データのパスとラベル情報をセットにして格納
            test_path_list.append((root_path + f"/test_images/{row[0]}", None))
    
    if disable_val:
        return path_list, test_path_list
    else:  
        train_path_list, val_path_list = train_test_split(path_list, test_size=test_size, random_state=random_seed)

        return train_path_list, val_path_list, test_path_list

class LemonDataset(data.Dataset):
    """
    レモンの画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path, img_label = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])
        img_filename = os.path.basename(img_path)
        
        
        if self.phase == 'test':
            return img_transformed, img_filename
        else:
            img_label = torch.tensor(img_label)
            return img_transformed, img_label, img_filename
    

def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.tensor(np.random.beta(alpha, alpha))
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(inputs, targets, beta=0):
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inputs.size()[0]).cuda()
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    return inputs, target_a, target_b, lam
    # compute output
    output = model(input)

def cutmix_criterion(criterion, output, target_a, target_b, lam):
    return criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)


class UnNormalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
    
from albumentations.core.transforms_interface import ImageOnlyTransform

def resize_square(img):
    """長辺のサイズで正方形の画像に"""
    l=max(img.shape[:2])
    
    h,w = img.shape[:2]
    hm = (l-h)//2
    wm = (l-w)//2
    return cv2.copyMakeBorder(img,
                            hm,
                            hm+(l-h)%2,
                            wm,
                            wm+(l-w)%2,
                            cv2.BORDER_CONSTANT,
                            value=0)

class CropLemon(ImageOnlyTransform):
    """レモンが写っている部分をcrop"""

    def __init__(self, margin=10, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.margin = margin

    def get_box(self, img):
        """ 中央に近い黄色い領域を見つける """
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        # h,v のしきい値で crop
        _, img_hcrop = cv2.threshold(h, 0, 40, cv2.THRESH_BINARY)
        _, img_vcrop = cv2.threshold(v, v.mean(), 255, cv2.THRESH_BINARY)
        th_img = (img_hcrop * (img_vcrop / 255)).astype(np.uint8)

        contours, hierarchy = \
            cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # サイズの大きいものだけ選択
        contours = [c for c in contours if cv2.contourArea(c) > 10000]
        if not contours: return None

        # 中央に近いものを選択
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])  # w, h
        min_contour = None
        min_dist = 1e10

        for c in contours:
            tmp = np.array(c).reshape(-1, 2)
            m = tmp.mean(axis=0)
            dist = sum((center - m) ** 2)
            if dist < min_dist:
                min_contour = tmp
                min_dist = dist

        box = [
            *(min_contour.min(axis=0) - self.margin).astype(np.int).tolist(),
            *(min_contour.max(axis=0) + self.margin).astype(np.int).tolist()]
        for i in range(4):
            if box[i] < 0: box[i] = 0
            if i % 2 == 0:
                if box[i] > img.shape[1]: box[i] = img.shape[1]
            else:
                if box[i] > img.shape[0]: box[i] = img.shape[0]

        return box  # left, top, right, bottom

    def apply(self, image, **params):
        image = image.copy()
        box = self.get_box(image)
        crop_img = None
        if not box or (box[3] - box[1] < 50 or box[2] - box[0] < 50):
            pass
        else:
            try:
                crop_img = image[box[1]:box[3], box[0]:box[2]]
            except:
                pass
        if crop_img is None:
            crop_img = image[40:, 10:-20]
        return resize_square(crop_img)

    def get_transform_init_args_names(self):
        return ("margin",)