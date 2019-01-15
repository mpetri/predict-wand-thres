#!/bin/bash

DEVICE=$1
QUANT=$2

python ../train.py --queries ../data/gov2_new/train-5k.json --dev_queries ../data/gov2_new/dev.json --device $1 --layers 2 --embed_size 32 --quantile $2
