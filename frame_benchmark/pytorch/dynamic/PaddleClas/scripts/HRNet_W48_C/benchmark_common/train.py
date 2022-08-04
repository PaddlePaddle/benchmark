# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    # Added code for distributed training
    parser.add_argument('--distributed', action='store_true', help='distributed training')
    # Added code for distributed training
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    # Added code for distributed training
    if args.distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(int(os.environ["LOCAL_RANK"]))
    else:
        local_rank = 0
    if local_rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))
    # Added code for distributed training

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    # Added code for distributed training
    if local_rank == 0:
        logger.info(get_model_summary(model, dump_input))
    # Added code for distributed training

    # copy model file
    # Added code for distributed training
    if local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if args.distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", local_rank)
    # Added code for distributed training
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    # Added code for distributed training
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda('cuda:{}'.format(local_rank))
    # Added code for distributed training

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Added code for distributed training
            if local_rank == 0:
                logger.info("=> loaded checkpoint (epoch {})"
                            .format(checkpoint['epoch']))
                best_model = True
            # Added code for distributed training
            
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
    traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # Added code for distributed training
    if args.distributed:
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        class CudaModel(torch.nn.Module):
            def __init__(self, model, device):
                super(CudaModel, self).__init__()
                self.model = model
                self.device = device
            
            def forward(self, input):
                with torch.no_grad():
                    input = input.to(self.device)
                return self.model(input)
        model = model.to(torch.device("cuda", local_rank))
        model = CudaModel(model, torch.device("cuda", local_rank))
    # creat distributed dataset
    if args.distributed:
        dist_sampler = DistributedSampler(train_dataset, shuffle=True)
        batch = config.TRAIN.BATCH_SIZE_PER_GPU
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch,
                                  sampler=dist_sampler,
                                  num_workers=config.WORKERS,
                                  pin_memory=True)
    # Added code for distributed training


    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # Added code for distributed training
        if args.distributed:
            dist_sampler.set_epoch(epoch)
        # Added code for distributed training
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    if local_rank == 0:
        final_model_state_file = os.path.join(final_output_dir,
                                            'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()


if __name__ == '__main__':
    main()