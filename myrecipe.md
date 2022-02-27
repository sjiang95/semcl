# MyRecipe

## Alienware

### resume

python3 main_moco.py -b 256 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=200 --start-epoch= --warmup-epochs=40 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=100 --resume  /media/quan/ContrastivePairs

### new

python3 main_moco.py -b 256 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=200 --warmup-epochs=40 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=100 /media/quan/ContrastivePairs

## dw505

python3 main_moco.py -b 512 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=200 --warmup-epochs=40 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=100 /mnt/SSD500G/quan/datasets/ContrastivePairs/
