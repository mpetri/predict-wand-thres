#!/bin/bash

COL=$1
DEVICE=$2
LAYERS=$3
QUANT=$4
TOPK=$5

python ../train.py --data_dir $COL \
                   --device $DEVICE \
                   --layers $LAYERS \
                   --batch_size 256 \
                   --k $TOPK \
                   --quantile $QUANT
