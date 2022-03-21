# MyRecipe

## Alienware

### resume

python3 main_moco.py -b 256 --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 --epochs=200 --start-epoch= --warmup-epochs=40 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=100 --resume  /media/quan/ContrastivePairs

### use pretrained 

python3 main_moco.py -b 128 --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 --epochs=100 --warmup-epochs=10 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=1 --choose-dataset='voc' /media/quan/ContrastivePairs
