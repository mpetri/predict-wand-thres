#!/bin/bash

COL=$1
DEVICE=$2

python ../train.py --data_dir $COL \ 
                   --device $DEVICE \
                   --layers 2 \
                   --embed_size 32 \
                   --quantile 0.99
