#!/bin/bash

COL=$1
DEVICE=$2
LAYERS=$3
QUANT=$4
BS=8192

python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size $BS \
                    --k 10 \
                    --quantile $QUANT

python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size $BS \
                    --k 100 \
                    --quantile $QUANT


python ../train.py --data_dir $COL \
                    --device $DEVICE \
                    --layers $LAYERS \
                    --batch_size $BS \
                    --k 1000 \
                    --quantile $QUANT

