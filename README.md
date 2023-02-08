# SemCL pretrain

## Introduction

This is a PyTorch implementation of SemCL for self-supervised ResNets and Swin Transformers. The following practice is tested on Ubuntu 20.04 LTS.

## Prerequisite

Clone and enter this repo.

```shell
git clone https://github.com/sjiang95/semclTraining
cd semclTraining
```

PyTorch [1.8.2 LTS](https://pytorch.org/get-started/previous-versions/#v182-with-lts-support)

```shell
conda create -n semcl pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda activate semcl
```

Other utils

```shell
conda install -c conda-forge prettytable torchinfo opencv grad-cam
```

where `prettytable` is used to formatting output during pretraining, `torchinfo` prints network structure in `moco2bkb.py`, and `opencv`&`grad-cam` are necessary for attention visualization.

## Usage: Preparation

PyTorch 1.8.2 LTS

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

## Usage: Self-supervised Pre-Training

Below are three examples for MoCo v3 pre-training.

### ResNet-50 with 2-node (16-GPU) training, batch 4096

On the first node, run:

```shell
python main.py \
  --moco-m-cos --crop-min=.2 \
  --dist-url 'tcp://[your first node address]:[specified port]' \
  --multiprocessing-distributed --world-size 2 --rank 0 \
  [your imagenet-folder with train and val folders]
```

On the second node, run the same command with `--rank 1`.
With a batch size of 4096, the training can fit into 2 nodes with a total of 16 Volta 32G GPUs.

On other nodes, run the same command with `--rank 1`, ..., `--rank 7` respectively.

### Notes

1. The batch size specified by `-b` is the total batch size across all GPUs.
2. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677) in [this line](https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L213).
3. Using a smaller batch size has a more stable result (see paper), but has lower speed. Using a large batch size is critical for good speed in TPUs (as we did in the paper).
4. In this repo, only *multi-gpu*, *DistributedDataParallel* training is supported; single-gpu or DataParallel training is not supported. This code is improved to better suit the *multi-node* setting, and by default uses automatic *mixed-precision* for pre-training.

## Citation
