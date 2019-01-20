#!/bin/bash

COL=$1
DEVICE=$2
LAYERS=$3

for QUANT in "0.5 0.7 0.9 0.99"
do
    python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size 256 \
                    --k 10 \
                    --quantile $QUANT

    python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size 256 \
                    --k 100 \
                    --quantile $QUANT


    python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size 256 \
                    --k 1000 \
                    --quantile $QUANT
done