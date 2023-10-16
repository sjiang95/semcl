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

```shell
conda create -n semcl pytorch=1.13 torchvision=0.14 pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate semcl
```

InfoNCE loss
We use the implementation of [RElbers/info-nce-pytorch](https://github.com/RElbers/info-nce-pytorch).

```shell
pip install info-nce-pytorch
```

Install `prettytable` to formatting output during pretraining.

```shell
conda install -c conda-forge prettytable
```

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
python main.py -a swin_tiny -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --dataset voc ade coco
```

For multi-node (let's say N nodes) training, on the first node, run:

```shell
python main.py -a swin_tiny -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://[your first node address]:[specified port]' --multiprocessing-distributed --world-size N --rank 0 --print-freq 1000 --dataset voc ade coco
```

From the second node, run the same command with `--rank 1` to `--rank N-1`.

### Notes

1. The batch size specified by `-b` is the total batch size across all GPUs.
2. The learning rate specified by `--lr` is the *base* lr, and is adjusted by the [linear lr scaling rule](https://arxiv.org/abs/1706.02677).
3. Using a smaller batch size has a more stable result (see paper), but has lower speed.

## Extract backbone

In MoCo framework we use, there are `base_encoder` and `momentum_encoder` in the saved pretrained model. It is the `base_encoder` that should be extracted for downstream tasks, including attention visualization. To this end, we provide a script `moco2bkb.py` to extract the `base_encoder`.

Install `torchinfo`

```shell
conda install -c conda-forge torchinfo
```

run

```shell
python moco2bkb.py -a swin_tiny /path/to/pretrained/checkpoint
```

The extracted backbone will be saved to `/path/to/pretrained/bkb_checkpoint`.

## Attention visualization

The script `visualize_attn.py`, together with the sample images in `visualize_attention/` we used in the paper, is provided for attention visualization. Install `opencv` and `grad-cam`

```shell
conda install -c conda-forge opencv grad-cam
```

then run

```shell
python visualize_attn.py -a swin_tiny --pretrained /path/to/pretrained/bkb_checkpoint --img-path visualize_attention/2008_000345.jpg
```

## Citation

```bibtex
@InProceedings{quan2023semantic,
    author    = {Quan, Shengjiang and Hirano, Masahiro and Yamakawa, Yuji},
    title     = {Semantic Information in Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {5686-5696}
}
```
