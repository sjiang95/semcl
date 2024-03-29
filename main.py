#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import shutil
import time
from datetime import datetime
import warnings
from functools import partial
import multiprocessing
from prettytable import PrettyTable
from utils.logging import get_logger

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
from torch.hub import load_state_dict_from_url

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
moco_m_global = None
iter_mode_global = None
forw_backw_iters = None
num_steps = None
parser = argparse.ArgumentParser(description='SemCL Pre-Training')
parser.add_argument('--dataroot', metavar='Path2ContrastivePairs', default='data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='swin_tiny',
                    type=str, choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: swin_tiny)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH2pretrainedWeights',
                    help="Path to pretrained weights having same architecture with --arch option.")
parser.add_argument('-j', '--workers', default=multiprocessing.cpu_count(), type=int, metavar='num_workers_per_gpu',
                    help='number of data loading workers (default: use multiprocessing.cpu_count() for every GPU)')
parser.add_argument('-e', '--epochs', default=None, type=int, metavar='num_epoch',
                    help='number of total epochs to run')
parser.add_argument('--iters', default=None, type=int, metavar='num_iter',
                    help='number of total iterations to update the model')
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
parser.add_argument('--lr', '--learning-rate', default=1.5e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar="moco's momentum",
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='weight_decay', help='weight decay (default: 0.1)',
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
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-ratio', default=None, type=float, metavar='warmup_ratio',
                    help='number of warmup iters=warmup_ratio*total_update_iter')
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


tb = PrettyTable(field_names=["key", "value"])


start_time = datetime.now()


def main():
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

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    assert args.distributed, "SyncBN depends on distributed training. If you really want to continue, comment out this line and take the risk of poor reults."

    ngpus_per_node = torch.cuda.device_count()

    # Retrieve pretrained weights
    if len(args.pretrained) == 0:
        args.pretrained = pretrained_weight_url[args.arch]
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
    global tb
    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    global iter_mode_global # Even global variables cannot share across processings. See https://stackoverflow.com/a/11215750. Solution is to set these vars inside each processing.
    if args.iters is not None and args.epochs is not None:
        raise AssertionError(
            "You can set either `--iters` or `--epochs`, not both.")
    elif args.iters is not None and args.epochs is None:
        iter_mode_global = 'iters'
    elif args.iters is None and args.epochs is not None:
        iter_mode_global = 'epochs'
    elif args.iters is None and args.epochs is None:
        args.iters = 30000
        print(
            f"Neither `--iters` nor `--epochs` is given, set total iterations to {args.iters}")
        iter_mode_global = 'iters'
    print(f"Iteration mode: {iter_mode_global}")

    if args.multiprocessing_distributed:
        print(f"Use GPU: {args.gpu} for printing")
    elif args.multiprocessing_distributed is False and args.gpu is not None:
        print(f"Use GPU: {args.gpu} for traning")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            tb.add_row(["DDP node rank", args.rank])
            args.rank = args.rank * ngpus_per_node + gpu
        print(
            f"[DDP] Attempt to initialize init_process_group() via {args.dist_url} with backend {args.dist_backend}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("[DDP] init_process_group() is initialized via backend:",
              dist.get_backend())
        dist.barrier()
        if args.gpu == 0:
            tb.add_row(["DDP world_size", dist.get_world_size()])

    if args.loss_mode == 'paired':
        print("Loss mode: pair-wise infoNCE")
    elif args.loss_mode == 'mocov3':
        print("Loss mode: mocov3(infoNCE)")
    elif len(args.loss_mode) == 0:
        print("Loss mode: test")
        args.loss_mode = 'test'
    else:
        raise ValueError("Unknown loss mode: ", args.loss_mode)
    tb.add_row(["Loss mode", args.loss_mode])

    # create model
    print(f"=> creating model '{args.arch}'")
    tb.add_row(["arch", args.arch])
    tb.add_row(["Initialize by", args.pretrained])
    if args.arch.startswith('swin'):
        # Unlike moco whose pretrained weights contain both base and momentum encoder,
        # swin transformer pretrained weights contains only the backbone (base encoder) itself.
        checkpoint = load_state_dict_from_url(args.pretrained, model_dir="pretrainedIN", map_location="cpu") if args.pretrained.startswith(
            "http") else torch.load(args.pretrained, map_location="cpu")
        state_dict_model = checkpoint["state_dict" if "state_dict" in checkpoint else "model"]
        for onekey in list(state_dict_model.keys()):  # remove "head"
            if onekey.startswith("head."):
                del state_dict_model[onekey]
        model = moco.builder.MoCo_Swin(
            partial(swin_transformer.__dict__[
                    args.arch], state_dict=state_dict_model),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, args.loss_mode
        )
        args.output_stride == None
    else:
        print(f"output_stride is {args.output_stride}.")
        tb.add_row(["output stride(os)", args.output_stride])
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
    tb.add_row(["softmax temperature", args.moco_t])

    # store total batch_size
    # equivalent equiv_batch_size=args.batch_size*grad_accum
    equiv_batch_size = args.batch_size*args.grad_accum
    tb.add_row(["batch size", equiv_batch_size])

    # infer learning rate before changing batch size
    tb.add_row(["user specified lr", args.lr])
    args.lr = args.lr * equiv_batch_size / 256
    tb.add_row(["scaled lr", args.lr])

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
    tb.add_rows([
        ["optimizer", args.optimizer],
        ["weight decay", args.weight_decay]
    ])

    scaler = torch.cuda.amp.GradScaler()

    list_datasets = args.dataset
    dataset_str = ""
    tb.add_row(["dataset(s)", list_datasets])
    for i, dataset in enumerate(list_datasets):
        if i == len(list_datasets)-1:
            dataset_str = dataset_str+dataset
        else:
            dataset_str = dataset_str+dataset+"N"

    cudnn.benchmark = True

    # Data loading code
    traindir = args.dataroot
    normalize = et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    print(f"Crop size: {args.cropsize}")
    tb.add_row(["crop size", args.cropsize])
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

    # use the following expression to remove tail mini-batch if len(train_loader) cannot be evenly divided by args.grad_accum
    iters_per_epoch = math.floor(
        len(train_loader)/args.grad_accum)*args.grad_accum

    global forw_backw_iters
    if args.epochs is None:
        # The total iterations should be extend for {args.grad_accum} times, since we do one optimizer step every {args.grad_accum} iters.
        # forward-backward iteration = update_iter*grad_accum
        forw_backw_iters = args.iters*args.grad_accum
        args.epochs = math.ceil(forw_backw_iters/iters_per_epoch)
    else:  # use the set epoch to calculate total iters
        # num epochs is unrelated to grad_accum. The model would be updated for args.epochs*iters_per_epoch/args.grad_accum times.
        forw_backw_iters = args.epochs*iters_per_epoch
    global num_steps
    num_steps = int(forw_backw_iters/args.grad_accum)
    if args.grad_accum > 1:
        print(
            f"Due to gradient accumulation {args.grad_accum}, the total forward-backward iteration is {forw_backw_iters} (~{args.epochs} epochs), which is equivalent to update the model for {num_steps} iterations as user specified with batchsize (grad_accum*batch_size=) {equiv_batch_size}).")
    tb.add_rows([
        ["epochs", args.epochs],
        ["iters(forward-backward)", forw_backw_iters],
        ["iters(update steps)", num_steps]
    ])

    # logger
    log_path = os.path.join("work_dirs",
                            dataset_str,
                            args.arch,
                            args.loss_mode,
                            f"batchsize{equiv_batch_size:04d}")
    log_file_name = os.path.join(log_path,
                                 f"{start_time.strftime('%Y%m%d_%H%M%S')}_{dataset_str}_{args.arch}{('os'+str(args.output_stride)) if args.output_stride is not None else ''}_{args.loss_mode}_ecd{args.epochs:04d}ep{(args.iters if args.epochs is None else num_steps)}itbatchsize{equiv_batch_size:04d}_crop{args.cropsize}.log")
    logger = get_logger(name="semclTraining", log_file=log_file_name)

    summary_writer = SummaryWriter(
        log_dir=os.path.join("work_dirs",
                             dataset_str,
                             args.arch,
                             args.loss_mode,
                             f"batchsize{equiv_batch_size:04d}",
                             ),
        filename_suffix=f"{args.arch}{('os'+str(args.output_stride)) if args.output_stride is not None else ''}_{args.loss_mode}_ecd{args.epochs:04d}ep{(args.iters if args.epochs is None else num_steps)}itbatchsize{equiv_batch_size:04d}_crop{args.cropsize}"
    ) if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank == 0) else None

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=num_steps, pct_start=0.125 if args.warmup_ratio is None else args.warmup_ratio)
    """
    [OneCycleLR — PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#onecyclelr)
    The total model update times is determined by args.iters regardless of args.grad_accum
    pct_start (float): The percentage of the cycle (in number of steps) spent increasing the learning rate.
    """

    # optionally resume from a checkpoint
    global moco_m_global
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(
                checkpoint["state_dict" if "state_dict" in checkpoint else "model"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            moco_m_global = checkpoint['moco_moemtum']
            logger.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            del checkpoint
            tb.add_row(["resume", args.resume])
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
    else:
        moco_m_global = args.moco_m
    tb.add_row(["moco momentum", moco_m_global])
    ckpt_path = os.path.join(args.output_dir, "ckpt",
                             dataset_str,
                             args.arch,
                             args.loss_mode,
                             f"batchsize{equiv_batch_size:04d}",
                             f"{dataset_str}_{args.arch}{('os'+str(args.output_stride)) if args.output_stride is not None else ''}_{args.loss_mode}_ecd{args.epochs:04d}ep{(args.iters if args.epochs is None else num_steps)}itbatchsize{equiv_batch_size:04d}_crop{args.cropsize}.pth.tar")
    logger.info(f"Checkpoints will be saved to {ckpt_path}.")
    tb.add_row(["save ckpt to", ckpt_path])
    logger.info(f"Training config summary:\n{tb}")

    # suppress printing if not first GPU on first node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    training_start_ts = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = datetime.now()
        logger.info(f"{epoch_start}: Start epoch {epoch}/{args.epochs}.")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, lr_scheduler,
              scaler, summary_writer, epoch, args, logger)

        # only the first GPU saves checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'moco_moemtum': moco_m_global,
            }, is_best=False, filename=ckpt_path
            )
            logger.info(
                f"{datetime.now()}: Save checkpoint of epoch {epoch} to {os.path.abspath(ckpt_path)}")

            epoch_end = datetime.now()
            logger.info(
                f"Epoch {epoch}/{args.epochs} takes {epoch_end-epoch_start}.")
            epoch_end_ts = time.time()
            # calculate ETA (estimated time of arrival)
            training_dur = epoch_end_ts-training_start_ts
            eta = datetime.fromtimestamp(time.time()+training_dur/(
                (epoch+1-args.start_epoch)*iters_per_epoch)*(forw_backw_iters-1-(epoch+1)*iters_per_epoch))
            logger.info(f"[ETA] {eta}, {time.tzname[0]}")
            print()

    if not args.multiprocessing_distributed or args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, lr_scheduler, scaler, summary_writer, epoch, args, logger):
    batch_time = AverageMeter('BatchTime', ':6.3f')
    data_time = AverageMeter('DataTime', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    moco_momentum = AverageMeter('moco_momemtum', ':.8e')
    progress = ProgressMeter(
        forw_backw_iters,  # num update iters is determined by user-defined iteration regardless of grad_accum. But forward-backward iteration is forw_backw_iters
        [batch_time, data_time, learning_rates, losses, moco_momentum],
        logger=logger,
        prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    # use condition i >= math.floor(iters_per_epoch/args.grad_accum)*args.grad_accum to remove tail mini-batch if iters_per_epoch cannot be evenly divided by args.grad_accum
    iters_per_epoch = math.floor(
        len(train_loader)/args.grad_accum)*args.grad_accum
    for i, (anchor_images, nanchor_images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # check iterations
        cur_iters = epoch*iters_per_epoch+i
        if (iter_mode_global == 'iters' and cur_iters >= args.iters) or i >= iters_per_epoch:
            progress.display(cur_iters)  # print status of last iteration
            break

        if args.gpu is not None:
            anchor_images = anchor_images.cuda(args.gpu, non_blocking=True)
            nanchor_images = nanchor_images.cuda(args.gpu, non_blocking=True)

        cur_lr = lr_scheduler.get_last_lr()[0]
        learning_rates.update(cur_lr)
        # adjust momentum coefficient per update iteration
        if (i+1) % args.grad_accum == 0 and args.moco_m_cos:
            moco_m_global = adjust_moco_momentum(
                epoch + i / iters_per_epoch, args)
        moco_momentum.update(moco_m_global)
        # compute output
        with torch.cuda.amp.autocast(True):
            # Normalize our loss (if averaged). See https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py-L5.
            loss = model(anchor_images, nanchor_images,
                         moco_m_global) / args.grad_accum
        losses.update(loss.item(), anchor_images[0].size(0))

        # compute gradient and do step
        # optimizer.zero_grad()                         # If we reset gradients tensors here, the gradients will never accumulate.
        scaler.scale(loss).backward()
        if (i+1) % args.grad_accum == 0:                # Wait for several backward steps
            # optimizer.step()                            # optimizer.step() should not be called when amp is applied. See https://discuss.pytorch.org/t/ddp-amp-gradient-accumulation-calling-optimizer-step-leads-to-nan-loss/162624.
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            # To avoid warning: Detected call of lr_scheduler.step() before optimizer.step(). In PyTorch 1.1.0 and later, you should call them in the opposite order: optimizer.step() before lr_scheduler.step(). Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
            # Solution: https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/10
            skip_lr_sched = (scale > scaler.get_scale())
            # Reset gradients tensors only if we have done a step. See https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3?permalink_comment_id=2921188#gistcomment-2921188. And https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation.
            optimizer.zero_grad()
            if args.rank == 0:
                summary_writer.add_scalar(
                    "loss", loss.item(), cur_iters)
                summary_writer.add_scalar(
                    "lr", cur_lr, cur_iters)
                summary_writer.add_scalar(
                    "moco_momentum", moco_m_global, cur_iters)
            if not skip_lr_sched:
                lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress at args.print_freq and epoch start
        if cur_iters % args.print_freq == 0 or i == 0 or i == iters_per_epoch-1:
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
    def __init__(self, num_batches, meters, logger, prefix="",):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch /
                    args.epochs)) * (1. - args.moco_m)
    return m


def load_moco_backbone(backbone: nn.Module, args):
    linear_keyword = 'fc'
    # load state_dict
    checkpoint = load_state_dict_from_url(args.pretrained, model_dir="pretrainedIN", map_location="cpu") if args.pretrained.startswith(
        "http") else torch.load(args.pretrained, map_location="cpu")
    if args.gpu == 0:
        tb.add_row([f"Initialize by", args.pretrained])
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
    print(
        f"Load pretrained {args.arch} weights from {args.pretrained} with unmatched keys: {msg}")
    return backbone


if __name__ == '__main__':
    main()
