
import math
import deepspeed
import torch.optim as optim

from adabelief_pytorch import AdaBelief

def make_optim(net, args):
    
    if args.phase == 'test':
        return net, None, None

    update_param_list = []
    

    if args.backbone == 'efficientnet-b0':
        if args.full_train:
            for name, param in net.named_parameters():
                param.requires_grad = True
            update_param_list.append({"params": net.parameters(), "lr": args.lr})
        else:
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in net.named_parameters():
                param.requires_grad = False

            print(f"--以下学習させる層の名前--")
            for name, param in net._blocks[-args.learn_depth:].named_parameters():
                param.requires_grad = True
                print(name)
                
            # 2. 最後のLayerモジュールを勾配計算ありに変更
            for name, param in net._conv_head.named_parameters():
                param.requires_grad = True
                print(name)

            # 3. 識別器を勾配計算ありに変更
            for name, param in net._fc.named_parameters():
                param.requires_grad = True
                print(name)
                
            update_param_list.append({"params": net._conv_head.parameters(), "lr": args.lr})
            update_param_list.append({"params": net._fc.parameters(), "lr": args.lr})
            update_param_list.append({"params": net._blocks[-args.learn_depth:].parameters(), "lr": args.lr})

        
    elif args.backbone == 'mobilenet_v2':
        
        if args.full_train:
            for name, param in net.named_parameters():
                param.requires_grad = True
            update_param_list.append({"params": net.parameters(), "lr": args.lr})
            
        else:
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in net.named_parameters():
                param.requires_grad = False

            print(f"--以下学習させる層の名前--")
            # 2. 最後のLayerモジュールを勾配計算ありに変更
            for name, param in net.features[-args.learn_depth:].named_parameters():
                param.requires_grad = True
                print(name)

            # 3. 識別器を勾配計算ありに変更
            for name, param in net.classifier.named_parameters():
                param.requires_grad = True
                print(name)
                
            update_param_list.append({"params": net.classifier.parameters(), "lr": args.lr})
            update_param_list.append({"params": net.features[-args.learn_depth:].parameters(), "lr": args.lr})

    elif args.backbone == 'resnet-152':
        if args.full_train:
            for name, param in net.named_parameters():
                param.requires_grad = True
            update_param_list.append({"params": net.parameters(), "lr": args.lr})
        else:
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in net.named_parameters():
                param.requires_grad = False

            print(f"--以下学習させる層の名前--")
            # 2. 最後のLayerモジュールを勾配計算ありに変更
            for name, param in net.layer4.named_parameters():
                param.requires_grad = True
                print(name)

            # 3. 識別器を勾配計算ありに変更
            for name, param in net.fc.named_parameters():
                param.requires_grad = True
                print(name)
                
            update_param_list.append({"params": net.fc.parameters(), "lr": args.lr})
            update_param_list.append({"params": net.layer4.parameters(), "lr": args.lr})

    elif args.backbone == 'resnet-18':
        if args.full_train:
            for name, param in net.named_parameters():
                param.requires_grad = True
            update_param_list.append({"params": net.parameters(), "lr": args.lr})
        else:
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in net.named_parameters():
                param.requires_grad = False

            print(f"--以下学習させる層の名前--")
            # 2. 最後のLayerモジュールを勾配計算ありに変更
            for name, param in net.layer4.named_parameters():
                param.requires_grad = True
                print(name)

            # 3. 識別器を勾配計算ありに変更
            for name, param in net.fc.named_parameters():
                param.requires_grad = True
                print(name)
                
            update_param_list.append({"params": net.fc.parameters(), "lr": args.lr})
            update_param_list.append({"params": net.layer4.parameters(), "lr": args.lr})

    elif args.backbone == 'densenet-161':
        if args.full_train:
            for name, param in net.named_parameters():
                param.requires_grad = True
            update_param_list.append({"params": net.parameters(), "lr": args.lr})
            
        else:
            # 1. まず全部を、勾配計算Falseにしてしまう
            for name, param in net.named_parameters():
                param.requires_grad = False

            print(f"--以下学習させる層の名前--")
            # 2. 最後のLayerモジュールを勾配計算ありに変更
            for name, param in net.features.denseblock4.named_parameters():
                param.requires_grad = True
                print(name)

            # 3. 識別器を勾配計算ありに変更
            for name, param in net.classifier.named_parameters():
                param.requires_grad = True
                print(name)
                
            update_param_list.append({"params": net.classifier.parameters(), "lr": args.lr})
            update_param_list.append({"params": net.features.denseblock4.parameters(), "lr": args.lr})
        
        
        
    # 最適化手法の設定
    if args.optim == 'sgd':
        optimizer = optim.SGD(params=update_param_list,momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(update_param_list,betas=(args.adam_beta_1, args.adam_beta_2),)
    elif args.optim == 'adaBelief':
        optimizer = AdaBelief(update_param_list,eps=1e-12, betas=(args.adam_beta_1,args.adam_beta_2), print_change_log=False) # epsilon は小さい方が良い（論文より）
    
    
    if args.is_scheduler:
        print('>>> use scheduler <<<')
        # スケジューラーの設定
        if args.scheduler == 'lambda':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: args.scheduler_lambda_param ** epoch)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_step_gamma)
        elif args.scheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_multistep_milestone, gamma=args.scheduler_multistep_gamma)
        elif args.scheduler == 'exponetial':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True, min_lr=1e-5)

        if args.use_zero_offload:
            parameters = filter(lambda p: p.requires_grad, net.parameters())
            net, optimizer, _, _= deepspeed.initialize(args=args, model=net, optimizer=optimizer, lr_scheduler=scheduler ,model_parameters=parameters)

        
        return net, optimizer, scheduler
    
    else:
        if args.use_zero_offload:
            parameters = filter(lambda p: p.requires_grad, net.parameters())
            net, optimizer, _, _= deepspeed.initialize(args=args, model=net, optimizer=optimizer,model_parameters=parameters)
            
        return net, optimizer, None