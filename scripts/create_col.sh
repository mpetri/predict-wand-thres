#!/bin/bash

INPUT_FILE=$1
COL_DIR=$2

echo "create col dir $COL_DIR"
mkdir -p $COL_DIR/models/

echo "copy input to dir $COL_DIR"
cp $INPUT_FILE $COL_DIR/raw_input.json

echo "random shuffle input"
shuf $COL_DIR/raw_input.json > $COL_DIR/shuf_input.json

NUM_TEST=1000
NUM_DEV=1000
NUM_DEV_AND_TEST=2000

echo "create dev/test/train files"
tail -n $NUM_DEV_AND_TEST $COL_DIR/shuf_input.json > $COL_DIR/dev_and_test.json

head -n $NUM_DEV $COL_DIR/dev_and_test.json > $COL_DIR/dev.json

tail -n $NUM_TEST $COL_DIR/dev_and_test.json > $COL_DIR/test.json

head -n -$NUM_DEV_AND_TEST $COL_DIR/shuf_input.json > $COL_DIR/train.json

head -n 5000 $COL_DIR/shuf_input.json > $COL_DIR/debug.json

NUM_QUERIES=$(wc -l $COL_DIR/shuf_input.json)
NUM_TRAIN=$(wc -l $COL_DIR/train.json)
NUM_DEV=$(wc -l $COL_DIR/dev.json)
NUM_TEST=$(wc -l $COL_DIR/test.json)
NUM_DEBUG=$(wc -l $COL_DIR/debug.json)

echo "NUM QUERIES = $NUM_QUERIES"
echo "NUM TRAIN = $NUM_TRAIN"
echo "NUM DEV = $NUM_DEV"
echo "NUM TEST = $NUM_TEST"
echo "NUM DEBUG = $NUM_DEBUG"