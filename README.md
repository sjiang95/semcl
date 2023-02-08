# SemCL pretrain

## Introduction

This is a PyTorch implementation of SemCL for self-supervised ResNets and Swin Transformers. The following practice is tested on Ubuntu 20.04 LTS.

## Prerequisite

Clone and enter this repo.

```shell
git clone https://github.com/sjiang95/semclTraining
cd semclTraining
```

We recommend `conda` for smooth experience.

```shell
conda env create -n semcl -f condaenvlinux.yml
conda activate semcl
```

Or, manually.

PyTorch [1.8.2 LTS](https://pytorch.org/get-started/previous-versions/#v182-with-lts-support)

```shell
conda create -n semcl pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda activate semcl
```

InfoNCE loss
We use the implementation of [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch).

```shell
pip install info-nce-pytorch
```

Other utils

```shell
conda install -c conda-forge prettytable torchinfo opencv grad-cam
```

where `prettytable` is used to formatting output during pretraining, `torchinfo` prints network structure in `moco2bkb.py`, and `opencv`&`grad-cam` are necessary for attention visualization.

## Dataset

Follow semclDataset (add link) to prepare the SemCL dataset, and create a symlink

```shell
ln -s /path/to/ContrastivePairs data
```

## Pretraining

Below are examples for SemCL pretraining.

### Swin-T, batch 64, SemCL-stuff

For single node training, run

```shell
python3 main.py -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --dataset voc ade coco
```

For multi-node (let's say N nodes) training, on the first node, run:

```shell
python3 main.py -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://[your first node address]:[specified port]' --multiprocessing-distributed --world-size N --rank 0 --print-freq 1000 --dataset voc ade coco
```

From the second node, run the same command with `--rank 1` to `--rank N-1`.

### Notes

1. The batch size specified by `-b` is the total batch size across all GPUs.
2. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677).
3. Using a smaller batch size has a more stable result (see paper), but has lower speed.

## Citation
