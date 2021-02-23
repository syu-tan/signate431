import torch
from torch import nn

def make_criterion(args):
    # 損失関数の設定

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.use_mse:
        criterion = nn.MSELoss()
    
    elif args.loss_weight:
        if args.loss_test_raito:
            weights = torch.tensor(args.loss_raito).to(device)
        else:
            # train_images.csv のラベル割合
            weights = torch.tensor([1/400, 1/255, 1/235, 1/212]).to(device)


        criterion = nn.CrossEntropyLoss(weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion
    