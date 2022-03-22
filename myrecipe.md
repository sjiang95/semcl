# My_Recipe

## VOC

python3 main_moco.py -b 128 --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 --epochs=100 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=1 --choose-dataset='voc' --loss-mode L /media/quan/ContrastivePairs

## ADE20K

python3 main_moco.py -b 128 --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 --epochs=200 --stop-grad-conv1 --moco-m-cos --moco-t=.2 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq=100 --choose-dataset='ade' --loss-mode L /media/quan/ContrastivePairs
