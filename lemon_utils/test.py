import csv

import numpy as np
from tqdm import tqdm

import torch


def test_model(nets, dataloaders_dict, args):
    
    print('>> test and make submit file')
    
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    if not isinstance(nets, list):
        nets = [nets]

    net = nets[0]
        
    # ネットワークをGPUへ
    net.to(device)
    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    
    results = list()
    
    net.eval()   # モデルを検証モードに

    # データローダーからミニバッチを取り出すループ
    for inputs, filenames in dataloaders_dict['test']:
        
        if args.use_zero_offload:
            inputs = inputs.to(device).half()
        else:
            inputs = inputs.to(device)

        # 順伝搬（forward）計算
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            
            if args.use_mse:
                preds = torch.tensor(np.clip(outputs.detach().cpu().numpy().round(),0,3).astype(np.int))
            else:
                _, preds = torch.max(outputs, 1)  # ラベルを予測
            preds_list = preds.cpu().tolist()

        for filename, pred in zip(filenames, preds_list):
            results.append([filename, pred])
                
    # 提出用ファイルの作成
    with open(f'{args.log_dir}submit.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    print(f'推論終了　＝＝＞　{args.log_dir}submit.csv')
                
                
    

