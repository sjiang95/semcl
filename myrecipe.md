# My_Recipe

## VOC

```bash
python3 main_moco.py -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --dataset voc --loss-mode L0 [--output-dir] [/path/to/Contrastivepairs]
```

## ADE20K

```bash
python3 main_moco.py -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --dataset ade --loss-mode L0 [--output-dir] [/path/to/Contrastivepairs]
```

## four datasets

```bash
python3 main_moco.py -a [arch] -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://192.168.1.1:10001' --multiprocessing-distributed --world-size 2 --rank [num_rank] --print-freq 1000 --dataset voc city ade coco --loss-mode L0 [--output-dir] [/path/to/Contrastivepairs]
```

python3 main_moco.py -b 64 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 1000 --dataset ade --loss-mode L0 [--output-dir] /media/quan/datasets/ContrastivePairs
