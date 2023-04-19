from __future__ import division
import argparse
import importlib
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openmixup import __version__
from openmixup.apis import init_random_seed, set_random_seed, train_model
from openmixup.datasets import build_dataset
from openmixup.models import build_model
from openmixup.utils import (collect_env, get_root_logger, traverse_replace,
                             setup_multi_processes)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# =========== optional arguments ===========
# --work-dir        存储日志和模型的目录
# --resume-from     加载 checkpoint 的目录
# --no-validate     是否在训练的时候进行验证
# 互斥组：
#   --gpus          使用的 GPU 数量
#   --gpu_ids       使用指定 GPU 的 id
# --seed            随机数种子
# --deterministic   是否设置 cudnn 为确定性行为
# --options         其他参数
# --launcher        分布式训练使用的启动器，可以为：['none', 'pytorch', 'slurm', 'mpi']
#                   none：不启动分布式训练，dist_train.sh 中默认使用 pytorch 启动。
# --local_rank      本地进程编号，此参数 torch.distributed.launch 会自动传入。
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config', 
        default='configs/classification/Mydata/automix/basic/new_r18_l2_a2_near_L1_01_mlr5e_2.py',
        help='train config file path')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file for backbone')
    parser.add_argument(
        '--load_checkpoint', default=None, help='load the checkpoint file of full models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int, default=1,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu_ids',
        type=int, nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int, default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    #args, unknown = parser.parse_known_args()
    args = parser.parse_args(args=[])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    # 设置 cudnn_benchmark = True 可以加速输入大小固定的模型. 如：SSD300
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    # work_dir 的优先程度为: 命令行 > 配置文件
    # 当 work_dir 为 None 的时候, 使用 ./work_dir/配置文件名 作为默认工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # os.path.basename(path)    返回文件名
        # os.path.splitext(path)    分割路径, 返回路径名和文件扩展名的元组
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    # 是否继续上次的训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    # 如果 launcher 为 none，不启用分布式训练。不使用 dist_train.sh 默认参数为 none
    if args.launcher == 'none':
        distributed = False
    # launcher 不为 none，启用分布式训练。使用 dist_train.sh，会传 ‘pytorch’
    else:
        distributed = True
        # 初始化 dist 里面会调用 init_process_group
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # 保存 config
    #cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'train_{}.log'.format(timestamp))
    # 获取 root logger。
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.pretty_text))

    # set random seeds
    # 设置随机化种子
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info('Set random seed to {}, deterministic: {}'.format(
        seed, args.deterministic))
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # build the model and load pretrained or checkpoint
    if args.pretrained is not None:
        assert isinstance(args.pretrained, str) and args.load_checkpoint is None
        cfg.model.pretrained = args.pretrained
        if cfg.model.get('init_cfg', None) is not None:
            cfg.model.init_cfg = None  # only initialize the backbone with args.pretrained
    # 构建模型: 需要传入 cfg.model, ?cfg.train_cfg, cfg.test_cfg
    model = build_model(cfg.model)
    if args.load_checkpoint is not None:
        assert isinstance(args.load_checkpoint, str) and args.pretrained is None
        load_checkpoint(model, args.load_checkpoint, map_location='cpu')

    print(cfg.data.train)
    # 构建数据集: 需要传入 cfg.data.train
    datasets = [build_dataset(cfg.data.train)]
    
    
    
    assert len(cfg.workflow) == 1, "Validation is called by hook."
    if cfg.checkpoint_config is not None:
        # save openmixup version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            openmixup_version=__version__, config=cfg.pretty_text)


    # 训练检测器, 传入：模型, 数据集, config 等
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
