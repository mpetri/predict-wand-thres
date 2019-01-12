#!/bin/bash

INPUT_FILE=$1
COL_DIR=$2

mkdir -p $COL_DIR

cp $INPUT_FILE $COL_DIR/raw_input.json

sort -R $COL_DIR/raw_input.json > $COL_DIR/shuf_input.json


NUM_TEST=1000
NUM_DEV=1000
NUM_DEV_AND_TEST=2000

tail -n $NUM_DEV_AND_TEST $COL_DIR/shuf_input.json > $COL_DIR/dev_and_test.json

head -n $NUM_DEV $COL_DIR/dev_and_test.json > $COL_DIR/dev.json

tail -n $NUM_TEST $COL_DIR/dev_and_test.json > $COL_DIR/test.json

head -n -$NUM_DEV_AND_TEST $COL_DIR/shuf_input.json > $COL_DIR/train.json

NUM_TRAIN=$(wc -l $COL_DIR/train.json)
NUM_DEV=$(wc -l $COL_DIR/dev.json)
NUM_TEST=$(wc -l $COL_DIR/test.json)

echo "NUM QUERIES = $NUM_QUERIES"
echo "NUM TRAIN = $NUM_TRAIN"
echo "NUM DEV = $NUM_DEV"
echo "NUM TEST = $NUM_TEST"