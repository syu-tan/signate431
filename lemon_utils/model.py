#!/usr/bin/env python 3
# coding: utf-8

import torch
from torch import nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet

def make_model(args, ensemble=False, backbone="efficientnet-b0"):
    
    if (ensemble and backbone == 'efficientnet-b0') or (not ensemble and args.backbone == 'efficientnet-b0'):
        # https://github.com/lukemelas/EfficientNet-PyTorch
        net = EfficientNet.from_pretrained('efficientnet-b0')
        net._fc = nn.Linear(in_features=1280, out_features=4)
    
    elif (ensemble and backbone == 'mobilenet_v2') or (not ensemble and args.backbone == 'mobilenet_v2'):
        net = models.mobilenet_v2(pretrained=True)
        net.classifier = nn.Linear(in_features=1280, out_features=4)

    elif (ensemble and backbone == 'resnet-152') or (not ensemble and args.backbone == 'resnet-152'):
        net = models.resnet152(pretrained=True)
        net.fc = nn.Linear(in_features=2048, out_features=4)

    elif (ensemble and backbone == 'resnet-18') or (not ensemble and args.backbone == 'resnet-18'):
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(in_features=512, out_features=4)

    elif (ensemble and backbone == 'densenet-161') or (not ensemble and args.backbone == 'densenet-161'):
        net = models.densenet161(pretrained=True)
        net.classifier = nn.Linear(in_features=2208, out_features=4)

    if args.use_mse:
        net = MSEModel(net, 4)

    if args.debug:
        print(net)
        
    return net

class EnsembleModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        models = []

        for config in args.ensemble_list:
            config_list = config.split(":")
            model = make_model(args, ensemble=True, backbone=config_list[0])
            weights = torch.load(config_list[1])
            model.load_state_dict(weights)
            models.append(model)

        self.models = nn.ModuleList(models)


    def forward(self, x):

        xs = []

        for i in range(len(self.models)):
            xs.append(self.models[i](x))

        x = torch.stack(xs, dim=1)
        x = x.mean(dim=1)

        return x
    
class MSEModel(nn.Module):
    def __init__(self, base_model, n_class):
        super().__init__()
        self.model = base_model
        self.softmax = nn.Softmax(dim=1)
        self.label_vals = torch.arange(n_class)

    def forward(self, x):
        x = self.model(x)
        return (self.softmax(x) * self.label_vals).sum(axis=1)

    def to(self, device, *args, **kwargs):
        self.label_vals = self.label_vals.to(device)
        return super().to(device, *args, **kwargs)


    