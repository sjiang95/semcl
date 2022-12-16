#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta
import warnings
from functools import partial
import multiprocessing
import requests

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer

import swin_transformer

from semclData import semclDataset
from utils import ext_transforms as et


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))

model_names = ['swin_tiny', 'swin_small', 'swin_base',
               'swin_large'] + torchvision_model_names

pretrained_weight_url = {
    'resnet50': 'https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar',
    'resnet101': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
    'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    'swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
    'swin_base': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
}

parser = argparse.ArgumentParser(description='SemCL Pre-Training')
parser.add_argument('--dataroot', metavar='Path2ContrastivePairs', default='data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    type=str, choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH2pretrainedWeights',
                    help="Path to pretrained weights having same architecture with --arch option.")
parser.add_argument('-j', '--workers', default=multiprocessing.cpu_count(), type=int, metavar='num_workers_per_gpu',
                    help='number of data loading workers (default: use multiprocessing.cpu_count() for every GPU)')
parser.add_argument('--epochs', default=None, type=int, metavar='num_epoch',
                    help='number of total epochs to run')
parser.add_argument('--iters', default=None, type=int, metavar='num_iter',
                    help='number of total iterations to update the model')
parser.add_argument('--iter-mode', type=str,
                    help='Iteration mode: total iters or total epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='start_epoch',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--cropsize', default=224, type=int, metavar='cropsize',
                    help='image crop size. Swin dopted 224*224.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='batchsize',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--grad-accum', default=1, type=int,
                    help='accumulation steps. Equivalent batch size would be batch_size*grad_accum.')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar="moco's momentum",
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='weight_decay', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='print_frequency', help='print frequency (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH2checkpoint',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-iters', default=None, type=int, metavar='warmup_iters',
                    help='number of warmup iters')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# choose datasets to load
parser.add_argument('--dataset', default=['coco', 'ade', 'voc'], nargs='+',
                    help='arbitrary combine coco, ade20k and voc2012 datasets')

# choose negative mode
parser.add_argument('--loss-mode', default='paired', type=str,
                    choices=['paired', 'mocov3'],
                    help='Determines how the (optional) negative_keys are handled. Value must be one of ["paired", "mocov3"], where the latter means original infoNCE loss.')

# set checkpoints output dir
parser.add_argument('--output-dir', default='.', type=str,
                    help='Output path. Default is current path.')

# deeplab
parser.add_argument("--output-stride", default=None, choices=[None, 8, 16],
                    help="This option is valid for only resnet backbones.")


def main():
    start_time = datetime.now()
    print(f"{start_time.strftime('%Y/%m/%d %H:%M:%S.%f')}: Training started.")
    print(f"Use Pytorch {torch.__version__} with cuda {torch.version.cuda}")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.iters is not None and args.epochs is not None:
        raise AssertionError(
            "You can set either `--iters` or `--epochs`, not both.")
    elif args.iters is not None and args.epochs is None:
        args.iter_mode = 'iters'
    elif args.iters is None and args.epochs is not None:
        args.iter_mode = 'epochs'
    elif args.iters is None and args.epochs is None:
        args.iters = 30000
        print(
            f"Neither `--iters` nor `--epochs` is given, set total iterations to {args.iters}")
        args.iter_mode = 'iters'

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    assert args.distributed, "SyncBN depends on distributed training. If you really want to continue, comment out this line and take the risk of poor reults."

    ngpus_per_node = torch.cuda.device_count()

    # Retrieve pretrained weights
    if len(args.pretrained) == 0:
        pretrained_weights_filename = pretrained_weight_url.copy()
        for one_key, one_value in pretrained_weights_filename.items():
            pretrained_weights_filename[one_key] = str(
                one_value).split(sep='/')[-1]

        path_to_pretrained_weights = os.path.join(
            'pretrained', pretrained_weights_filename[args.arch])
        if not os.path.exists('pretrained'):
            os.mkdir('pretrained')
            download_preweights(pretrained_weight_url,
                                path_to_pretrained_weights, args.arch)
        else:
            # Download dict file if not exists
            if not os.path.exists(path_to_pretrained_weights):
                download_preweights(pretrained_weight_url,
                                    path_to_pretrained_weights, args.arch)
        args.pretrained = path_to_pretrained_weights
        print("Use pretrained weight at", args.pretrained)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        args.workers = args.workers//ngpus_per_node
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    end_time = datetime.now()
    print(f"{end_time.strftime('%Y/%m/%d %H:%M:%S.%f')}: Training finished.")
    train_time_consume = end_time-start_time
    print("This training process takes ", str(train_time_consume))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.multiprocessing_distributed:
        print("Use GPU: {} for printing".format(args.gpu))
    elif args.multiprocessing_distributed is False and args.gpu is not None:
        print("Use GPU: {} for traning".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(
            f"[DDP] Attempt to initialize init_process_group() via {args.dist_url} with backend {args.dist_backend}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("[DDP] init_process_group() is initialized via backend:",
              dist.get_backend())
        dist.barrier()

    if args.loss_mode == 'paired':
        print("Loss mode: pair-wise infoNCE")
    elif args.loss_mode == 'mocov3':
        print("Loss mode: mocov3(infoNCE)")
    elif len(args.loss_mode) == 0:
        print("Loss mode: test")
        args.loss_mode = 'test'
    else:
        raise ValueError("Unknown loss mode: ", args.loss_mode)

    if args.output_dir == '.':
        print("Checkpoints will be written to current folder.")
    else:
        print("Checkpoints will be written to %s." % (args.output_dir))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('swin'):
        # Unlike moco whose pretrained weights contain both base and momentum encoder,
        # swin transformer pretrained weights contains only the backbone (base encoder) itself.
        model = moco.builder.MoCo_Swin(
            partial(swin_transformer.__dict__[
                    args.arch], pretrained=args.pretrained),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, args.loss_mode
        )
        args.output_stride == None
    else:
        print(f"output_stride is {args.output_stride}.")
        # from deeplabv3plus. See https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/4e1087de98bc49d55b9239ae92810ef7368660db/network/modeling.py#L34.
        if args.output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
        elif args.output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        # default resnet. See https://github.com/pytorch/vision/blob/5b4f79d9ba8cbeeb8d6f0fbba3ba5757b718888b/torchvision/models/resnet.py#L186.
        elif args.output_stride == None:
            replace_stride_with_dilation = None
        else:
            raise ValueError(
                f"The options '--output-stride' support only None, 8 or 16, but got {args.output_stride} of type {type(args.output_stride)}.")
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[
                    args.arch], zero_init_residual=True, replace_stride_with_dilation=replace_stride_with_dilation),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, args.loss_mode)
        model = load_moco_backbone(model, args=args)

    # store total batch_size
    # equivalent equiv_batch_size=args.batch_size*grad_accum
    equiv_batch_size = args.batch_size*args.grad_accum

    # infer learning rate before changing batch size
    args.lr = args.lr * equiv_batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError(
            "You have to either use DistributedDataParallel or specify one GPU by `--gpu`.")

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()

    list_datasets = args.dataset
    dataset_str = ""
    for i, dataset in enumerate(list_datasets):
        if i == len(list_datasets)-1:
            dataset_str = dataset_str+dataset
        else:
            dataset_str = dataset_str+dataset+"N"

    summary_writer_str = ('%s_%s_%s_batchsize%04d' % (
        dataset_str, args.arch, args.loss_mode, equiv_batch_size))
    summary_writer = SummaryWriter(comment=summary_writer_str, filename_suffix=summary_writer_str) if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank == 0) else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.dataroot
    normalize = et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    print(f"Crop size: {args.cropsize}")
    # directly resize original image for complete semantic information
    augmentation0 = [
        et.ExtResize(args.cropsize),
        et.ExtRandomCrop(args.cropsize),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        normalize
    ]

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        # et.ExtResize(args.cropsize),
        # et.ExtRandomCrop(args.cropsize),
        et.ExtRandomResizedCrop(args.cropsize, scale=(args.crop_min, 1.)),
        et.ExtRandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        et.ExtRandomGrayscale(p=0.2),
        et.ExtRandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        normalize
    ]

    augmentation2 = [
        # et.ExtResize(args.cropsize),
        # et.ExtRandomCrop(args.cropsize),
        et.ExtRandomResizedCrop(args.cropsize, scale=(args.crop_min, 1.)),
        et.ExtRandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        et.ExtRandomGrayscale(p=0.2),
        et.ExtRandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        et.ExtRandomApply([moco.loader.Solarize()], p=0.2),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        normalize
    ]

    train_dataset = semclDataset(traindir,
                                 transform=moco.loader.TwoCropsTransformWithItself(
                                     et.ExtCompose(augmentation0),
                                     et.ExtCompose(augmentation1),
                                     et.ExtCompose(augmentation2)),
                                 datasets=list_datasets
                                 )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    print("Use %d workers in torch.utils.data.DataLoader for each GPU." %
          args.workers)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    iters_per_epoch = len(train_loader)

    # The total epochs (iterations) should be extend for {args.grad_accum} times, since we do one optimizer step every {args.grad_accum} iters.
    if args.epochs is None:
        # forward-backward iteration = update_iter*grad_accum
        forw_backw_iters = args.iters*args.grad_accum
        args.epochs = math.ceil(forw_backw_iters/iters_per_epoch)
    else:  # use the set epoch to calculate total iters
        # num epochs is unrelated to grad_accum. The model would be updated for args.epochs*iters_per_epoch/args.grad_accum times.
        forw_backw_iters = args.epochs*iters_per_epoch
    print(
        f"Model will be updated for {forw_backw_iters/args.grad_accum} iterations ({args.epochs} epochs).")

    if args.warmup_iters is None:
        args.warmup_iters = forw_backw_iters//8
        print("warmup_iters is not given. Set it to", args.warmup_iters)
    else:
        assert args.warmup_iters <= forw_backw_iters, f" Warmup iteration({args.warmup_iters}) must be smaller than total forward&backward iterations({forw_backw_iters})."
        print(f"User specified warmup_iters={args.warmup_iters}")

    training_start_ts = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = datetime.now()
        print(f"{epoch_start}: Start epoch {epoch}/{args.epochs}.")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer,
              scaler, summary_writer, epoch, args)

        # only the first GPU saves checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            ckpt_path = os.path.join(args.output_dir, "ckpt",
                                     dataset_str,
                                     args.arch,
                                     args.loss_mode,
                                     f"batchsize{equiv_batch_size:04d}",
                                     f"{dataset_str}_{args.arch}{('os'+str(args.output_stride)) if args.output_stride is not None else ''}_{args.loss_mode}_ecd{args.epochs:04d}ep{(args.iters if args.epochs is None else forw_backw_iters/args.grad_accum):05d}itbatchsize{equiv_batch_size:04d}_crop{args.cropsize}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=ckpt_path
            )
            print(
                f"{datetime.now()}: Save checkpoint of epoch {epoch} to {os.path.abspath(ckpt_path)}")

            epoch_end = datetime.now()
            print(
                f"Epoch {epoch}/{args.epochs} takes {epoch_end-epoch_start}.")
            epoch_end_ts = time.time()
            # calculate ETA (estimated time of arrival)
            training_dur = epoch_end_ts-training_start_ts
            eta = datetime.fromtimestamp(time.time()+training_dur/(
                (epoch+1-args.start_epoch)*iters_per_epoch)*(args.iters-1-(epoch+1)*iters_per_epoch))
            print(f"[ETA] {eta}, {time.tzname[0]}")
            print()

    if not args.multiprocessing_distributed or args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        args.iters,
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (anchor_images, nanchor_images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # check iterations
        cur_iters = epoch*iters_per_epoch+i
        if args.iter_mode == 'iters' and cur_iters >= args.iters:
            progress.display(cur_iters-1)  # print status of last iteration
            break

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, cur_iters, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            anchor_images = anchor_images.cuda(args.gpu, non_blocking=True)
            nanchor_images = nanchor_images.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            # Normalize our loss (if averaged). See https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py-L5.
            loss = model(anchor_images, nanchor_images,
                         moco_m) / args.grad_accum

        losses.update(loss.item(), anchor_images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar(
                "loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do step
        # optimizer.zero_grad()                         # If we reset gradients tensors here, the gradients will never accumulate.
        scaler.scale(loss).backward()
        if (i+1) % args.grad_accum == 0:                # Wait for several backward steps
            # optimizer.step()                            # optimizer.step() should not be called when amp is applied. See https://discuss.pytorch.org/t/ddp-amp-gradient-accumulation-calling-optimizer-step-leads-to-nan-loss/162624.
            scaler.step(optimizer)
            scaler.update()
            # Reset gradients tensors only if we have done a step. See https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3?permalink_comment_id=2921188#gistcomment-2921188. And https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation.
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cur_iters % args.print_freq == 0 or i == iters_per_epoch-1 or i == 0:
            progress.display(cur_iters)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    par_dir = os.path.dirname(filename)
    if os.path.exists(par_dir) is not True:
        os.makedirs(par_dir)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, iter, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if iter < args.warmup_iters:
        lr = args.lr * iter / args.warmup_iters
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (iter -
                              args.warmup_iters) / (args.iters - args.warmup_iters)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch /
                    args.epochs)) * (1. - args.moco_m)
    return m


def load_moco_backbone(backbone: nn.Module, args):
    linear_keyword = 'fc'
    # load state_dict
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    if args.arch == 'resnet50':
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            # and not k.startswith('module.base_encoder.%s' % linear_keyword)
            if k.startswith('module.'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    else:  # resnet101
        # rename pretained layers
        state_dict = checkpoint
        for k in list(state_dict.keys()):
            if k.startswith(linear_keyword):
                continue
            state_dict['base_encoder.'+k] = state_dict[k]
            state_dict['momentum_encoder.'+k] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    args.start_epoch = 0
    msg = backbone.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}, f"Missing keys: {msg.missing_keys}" # comment out this line to debug
    return backbone


def download_preweights(list_url, download_path, key):
    print("Download pretrained weights for %s backbone from '%s'." %
          (key, list_url[key]))
    down_res = requests.get(list_url[key])
    with open(download_path, 'wb') as file:
        file.write(down_res.content)
    print("Download pretrained weights for %s backbone is saved to '%s'." %
          (key, download_path))


if __name__ == '__main__':
    main()
