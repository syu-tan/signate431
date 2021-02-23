
from tqdm import tqdm

import numpy as np
import torch
import deepspeed
import tensorboardX as tbx

from lemon_utils.data import mixup_data, mixup_criterion, cutmix_data, cutmix_criterion, UnNormalizer

# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, scheduler, k_fold, args):
    
    # SummaryWriterのインスタンス作成
    writer = tbx.SummaryWriter(args.log_dir)
    
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    
    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    unorm = UnNormalizer(mean=args.mean, std=args.std)
    
    epochs = args.epochs
    val_acc_list =[]
    net_best = None
    val_loss_best = 1000.0


    # epochのループ
    for epoch in range(epochs):

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue
            # 検証なしの場合は省略
            elif args.disable_val and (phase == 'val'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels, filenames in dataloaders_dict[phase]:
    
                if args.use_zero_offload:
                    inputs = inputs.to(device).half()
                    labels = labels.to(device)
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)            
                    # optimizerを初期化
                    optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    r = np.random.rand(1)

                    if args.mixup and phase == 'train' and (epoch - epochs) > args.strong_aug_cooldown_epoch and r < args.mixup_p:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.cutmix_beta)
                        outputs = net(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                    elif args.cutmix and phase == 'train' and (epoch - epochs) > args.strong_aug_cooldown_epoch and r < args.cutmix_p:
                        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, args.cutmix_beta)
                        outputs = net(inputs)
                        loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

                    else:
                        outputs = net(inputs)
                        
                    
                    if args.use_mse:
                        loss = criterion(outputs, labels.float())  
                        preds = torch.tensor(np.clip(outputs.detach().cpu().numpy().round(),0,3).astype(np.int)).to(device)
                    else:
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # モデル学習後の不正解を可視化
                    if (epoch+1) == epochs and phase == 'val':
                        for false_data in make_misstake_label_list(inputs=inputs, labels=labels, preds=preds, filenames=filenames):
                            writer.add_image(f'label:{false_data[1]}/filename:{false_data[3]} predict:{false_data[2]} k_fold:{k_fold}', unorm(false_data[0]), epoch)
  
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        if args.use_zero_offload:
                            net.backward(loss)
                            net.step()

                        else:
                            loss.backward()
                            optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)  
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            if args.is_k_fold or args.best_model:
                # 最高のモデルを保存する
                if val_loss_best > epoch_loss and phase == 'val':
                    net_best = net.cpu()
                    val_loss_best = epoch_loss
                    torch.save(net.state_dict(), args.save_dir + f"best_k_fold{k_fold}_{epoch}epoch.pth")



            # 最適化schedulerの更新
            if (args.is_scheduler and phase == 'train') or (args.scheduler == 'plateau' and phase == 'val'):
                if args.scheduler == 'plateau':
                    # 検証時の精度向上がなければ更新
                    scheduler.step(epoch_acc)
                else:
                    scheduler.step()
                
            
            # tensorboard用log出力設定
            if phase == "train":
                if k_fold >= 0:
                    writer.add_scalar(f"loss/train-k_fold:{k_fold}", epoch_loss, epoch)
                    writer.add_scalar(f"acc/train-k_fold:{k_fold}", epoch_acc, epoch)
                else:
                    writer.add_scalar(f"loss/train", epoch_loss, epoch)
                    writer.add_scalar(f"acc/train", epoch_acc, epoch)                   
            else:
                if k_fold >= 0:
                    writer.add_scalar(f"loss/val-k_fold:{k_fold}", epoch_loss, epoch)
                    writer.add_scalar(f"acc/val-k_fold:{k_fold}", epoch_acc, epoch)
                else:
                    writer.add_scalar(f"loss/val", epoch_loss, epoch)
                    writer.add_scalar(f"acc/val", epoch_acc, epoch)                
                val_acc_list.append(epoch_acc)
                
            if (epoch+1) % args.save_frequency == 0:
                # 中間でのモデルの保存
                if k_fold < 0:
                    if args.use_zero_offload:
                        net.save_checkpoint(args.save_dir, tag=None, client_state={}, save_latest=True)
                    else:
                        torch.save(net.state_dict(), args.save_dir + f"{epoch}epoch.pth")
                else:
                    torch.save(net.state_dict(), args.save_dir + f"k_fold{k_fold}_{epoch}epoch.pth")
                    

            if epoch % args.print_frequency == 0:
                print(f'k_fold:{k_fold} Epoch {epoch+1}/{epochs}', '{} Loss: {:.5f} Acc: {:.5f}'.format(phase, epoch_loss, epoch_acc))
    
    if not args.disable_val:            
        # 最適化のため検証用の精度
        max_acc = max(val_acc_list)
        print(f"k_fold:{k_fold} >>> max val acc : {max_acc}")
    
    # 試験の状態を保存
    if k_fold < 0:
        if args.use_zero_offload:
            torch.save(net.network.state_dict(), args.save_dir + "last.pth")
        else:
            torch.save(net.state_dict(), args.save_dir + "last.pth")
    else:
        torch.save(net.state_dict(), args.save_dir + f"k_fold{k_fold}_{epoch}epoch.pth")

    # writerを閉じる
    writer.close()

    if args.is_k_fold:
        return net_best
    elif args.best_model:
        return net_best
    else:
        return net


def make_misstake_label_list(inputs, labels, preds, filenames):
    """ 不正解データのリストの作成 """
    false_list = []
    for idx in range(len(labels)):

        if int(labels[idx]) == int(preds[idx]):
            # true_lst.append(lst)
            pass
        else:
            lst = [inputs[idx], labels[idx], preds[idx], filenames[idx]]
            false_list.append(lst)

    return false_list


