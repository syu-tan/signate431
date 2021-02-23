import os
import json
import argparse
import datetime

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(argv):
    """ 基本的な設定を管理する """
    parser = argparse.ArgumentParser('train lemon model' )
    # primary config
    parser.add_argument('--phase', type=str, default="train")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--num_works', type=int, default=16)
    parser.add_argument('--disable_val', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--backbone', type=str, default='mobilenet_v2')
    parser.add_argument('--best_model', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--is_k_fold', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--k_fold_num', type=int, default=4)
    parser.add_argument('--loss_weight', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--loss_test_raito', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--loss_raito', default=[1/714, 1/363, 1/296, 1/277])
    parser.add_argument('--use_mse', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_denoise', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--denoise_path', type=str, default='/home/syu/src/signate/431/searnch/noise_data.csv')
    



    # optimizer
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--learn_depth', type=int, default=3)
    parser.add_argument('--full_train', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--adam_beta_1', type=float, default=0.9)
    parser.add_argument('--adam_beta_2', type=float, default=0.999)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_weight_decay', type=float, default=0.0001)
    parser.add_argument('--is_scheduler', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--scheduler', type=str, default='multistep')
    parser.add_argument('--scheduler_lambda_param', type=float, default=0.99)
    parser.add_argument('--scheduler_step', type=int, default=20)
    parser.add_argument('--scheduler_step_gamma', type=float, default=0.7)
    parser.add_argument('--scheduler_multistep_milestone', default=[70, 90])
    parser.add_argument('--scheduler_multistep_gamma', type=float, default=0.5)
    
    # strong augmentation
    parser.add_argument('--strong_aug_cooldown_epoch', type=int, default=5)
    parser.add_argument('--mixup', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--mixup_p', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=1.0)
    parser.add_argument('--cutmix', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--cutmix_p', type=float, default=0.5)
    parser.add_argument('--cutmix_beta', type=float, default=0.5)


    # week augmentation
    parser.add_argument('--aug_croplemon', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_rotate', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_rorate_limit', type=int, default=89)
    parser.add_argument('--aug_rorate_p', type=float, default=0.8)
    parser.add_argument('--aug_random_gamma', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_random_gamma_limit', default=(65, 145))
    parser.add_argument('--aug_random_gamma_p', type=float, default=0.5)
    parser.add_argument('--aug_random_brightness', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_random_brightness_p', type=float, default=0.5)
    parser.add_argument('--aug_elastic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_elastic_p', type=float, default=0.5)
    parser.add_argument('--aug_blur', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_blur_p', type=float, default=0.5)
    parser.add_argument('--aug_cutout', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_cotout_p', type=float, default=0.5)
    parser.add_argument('--aug_downscale', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_downscale_min', type=float, default=0.10)
    parser.add_argument('--aug_downscale_max', type=float, default=0.35)
    parser.add_argument('--aug_downscale_p', type=float, default=0.5)
    parser.add_argument('--aug_fancy_pca', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_fancy_pca_p', type=float, default=0.5)
    parser.add_argument('--aug_gaussnoise', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_gaussnoise_p', type=float, default=0.5)
    parser.add_argument('--aug_random_contrast', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_random_contrast_p', type=float, default=0.5)
    parser.add_argument('--aug_random_shadow', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_random_shadow_p', type=float, default=0.5)
    parser.add_argument('--aug_random_resize_crop', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--aug_random_resize_crop_scale', default=(0.5, 1.1))
    parser.add_argument('--aug_random_resize_crop_ratio', default=(0.75, 1.3333333333333333))
    parser.add_argument('--aug_horizontalflip', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_horizontalflip_p', type=float, default=0.5)
    parser.add_argument('--aug_verticalflip', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--aug_verticalflip_p', type=float, default=0.5)
    parser.add_argument('--test_preprocess_centercrop', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--mean', default=(0.485, 0.456, 0.406))
    parser.add_argument('--std', default=(0.229, 0.224, 0.225))
    parser.add_argument('--size', type=int, default=224)

    # ensemble
    parser.add_argument('--is_ensemble', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--ensemble_list', default=['mobilenet_v2:/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/model/signate/431/20210215-1613354173/last.pth', 'mobilenet_v2:/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/model/signate/431/20210215-1613388475/last.pth'])

    # zero-offload
    parser.add_argument('--use_zero_offload', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--deepspeed_config', type=str, default='./lemon_utils/deep_speed_config.json')
    parser.add_argument('--with_cuda', default=True, action='store_true',help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',help='whether use exponential moving average')
    parser.add_argument('--local_rank', type=int, default=-1,help='local rank passed from distributed launcher')

    # python config
    parser.add_argument('--test_model_path', type=str, default='/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/model/signate/431/20210215-1613354173/last.pth')
    parser.add_argument('--root_path', type=str, default='/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/data/signate/431')
    parser.add_argument('--log_dir', type=str, default="./logs/")
    parser.add_argument('--save_dir', type=str, default="/media/syu/7d1c582d-4eb8-4836-9f6e-6a8bc8609413/model/signate/431/")
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--print_frequency', type=int, default=1)
    parser.add_argument('--save_frequency', type=int, default=100)
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False)
    # parser.add_argument('--gpu', type=str2bool, nargs='?', const=True, default=False)


    return parser.parse_args(argv)

def manage_log(args):
    """ ログをハイパーパラメーターと共に保存し、管理する """
    dir_name = datetime.datetime.today().strftime('%Y%m%d-%s') if not args.log_name else args.log_name
    args.save_dir = os.path.join(args.save_dir, dir_name) + "/"
    args.log_dir = os.path.join(args.log_dir, dir_name) + "/"
    check_dirs([args.save_dir, args.log_dir])

    with open(args.log_dir + 'args.json', 'wt') as f:
        json.dump(args.__dict__, f, indent=2)
        print(args.__dict__)

def check_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)