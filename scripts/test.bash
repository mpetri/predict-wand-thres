
MODEL=$1
DEVICE=$2

python ../test.py --queries ../data/gov2_new/test.json --model $MODEL --device $DEVICE
