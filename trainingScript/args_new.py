from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
import argparse
import os
import torch
import yaml
import shutil
from datetime import datetime
from utils.conf import set_random_seed

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, default='seq-cifar10',
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, default='er',
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name决定要不要使用自监督', choices=[None, 'simsiam' , 'barlowtwins'])
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=1, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='kaixuanzhang', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')



def get_args():
    parser = argparse.ArgumentParser()
    # 用config文件来存储配置信息，可以学习一下，之后的数据集和方法比较多
    parser.add_argument('-c', '--config-file', default='configs/simsiam_c10.yaml', type=str, help="xxx.yaml")
    # 提供时是True，不提供时是False
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='Data')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/cifar10_results/')
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    # eval形式
    parser.add_argument('--eval_from', type=str, default=None)
    # 是否显示进度条
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--ood_eval', action='store_true',
                        help='Test on the OOD set')
    parser.add_argument('--run', type=int, default=0, help='run')
    parser.add_argument('--pnn_base_widths', type=int, default=64, help='run')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval:
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]
    args.cl_type = 'scl' if args.cl_default else 'ucl'
    if args.pnn_base_widths != 64:
        pnn = '_pnn%d'%args.pnn_base_widths
    else:
        pnn = ''

    args.name = args.cl_type + '_' + args.name + '_' + args.model.cl_model + '_run_' + str(args.run)


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + datetime.now().strftime('%m%d%H%M%S_') + args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_random_seed(args.seed)


    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'image_size': args.dataset.image_size,
        'cl_default': args.cl_default
    }
    vars(args)['dataset_kwargs'] = {
        # 'name':args.model.name,
        # 'image_size': args.dataset.image_size,
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
        # 'drop_last': True,
        # 'pin_memory': True,
        # 'num_workers': args.dataset.num_workers,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args
