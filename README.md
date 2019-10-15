# PREDICT WAND THRESHOLDS

Prediction code for the paper

# USAGE

0. Install python requirements

```
pip install -r requirements.txt
```

1. Use `./tools/index_stats_json.cpp` to extract statistics for queries from an inverted index (pisa/ds2i):

```
./index_stats_json block_optpfor index_file qry_file wand_data > query_data.json
```

2. Create a collection using the `./scripts/create_col.sh` script:

```
./scripts/create_col.sh ./query_data.json ./data/sample/
create col dir ./data/sample/
copy input to dir ./data/sample/
random shuffle input
create dev/test/train files
NUM QUERIES = 3000 ./data/sample//shuf_input.json
NUM TRAIN = 1000 ./data/sample//train.json
NUM DEV = 1000 ./data/sample//dev.json
NUM TEST = 1000 ./data/sample//test.json
NUM DEBUG = 3000 ./data/sample//debug.json
```

which splits the input qry file into train/dev/test parts

3. To train the MLP model point it to the directory and specify some parameters (see paper for details)

```
python ./train.py --data_dir ./data/sample/ --device cuda:0 --layers 4 --batch_size 8192 --k 1000 --quantile 0.85
```

4. Make predictions on the test file:

```
python ./test.py --model ./data/sample/MLP-L4-Q0.85-K1000.model --queries ./data/sample/test.json --k 1000
Parameters:
        device: cpu
        k: 1000
        model: .\data\sample\models\MLP-L4-Q0.85-K1000.model
        queries: .\data\sample\test.json
Using torch device: cpu
read qrys: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 2946.22qrys/s]
skipped queries 1 out of 1000
dataset statistics: .\data\sample\test.json
        queries = 999
MLP(
  (layers): Sequential(
    (Dropout_1): Dropout(p=0.25, inplace=False)
    (linear_1): Linear(in_features=353, out_features=353, bias=True)
    (ReLU_1): ReLU()
    (BatchNorm1d_1): BatchNorm1d(353, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Dropout_2): Dropout(p=0.25, inplace=False)
    (linear_2): Linear(in_features=353, out_features=353, bias=True)
    (ReLU_2): ReLU()
    (BatchNorm1d_2): BatchNorm1d(353, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Dropout_3): Dropout(p=0.25, inplace=False)
    (linear_3): Linear(in_features=353, out_features=353, bias=True)
    (ReLU_3): ReLU()
    (BatchNorm1d_3): BatchNorm1d(353, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Dropout_4): Dropout(p=0.25, inplace=False)
    (linear_4): Linear(in_features=353, out_features=353, bias=True)
    (ReLU_4): ReLU()
    (BatchNorm1d_4): BatchNorm1d(353, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (last_linear): Linear(in_features=353, out_features=1, bias=True)
  )
)
11524960,3,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,1.3448028564453125,11.096502304077148
8392339,2,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,1.8396692276000977,11.810015678405762
8131844,3,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,0.5954630374908447,13.240103721618652
13543375,4,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,0.6387763619422913,24.392982482910156
5142267,2,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,0.5985416769981384,14.090593338012695
3572178,5,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,2.435288190841675,18.439430236816406
7839604,2,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,1.2566938400268555,9.851978302001953
11453547,5,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,1.8073967695236206,22.740262985229492
12095757,2,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,0.7809287905693054,18.283191680908203
8299525,5,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,1.9522461891174316,23.529333114624023
8888808,2,1000,.\data\sample\models\MLP-L4-Q0.85-K1000.model,0.1550535755913953,0.6513724327087402,12.432856559753418
...
```

where the columns are

```
qid,qlen,k,model_file,RHO,prediction,actual_threshold
```

so the actual prediction made by the model is `prediction`





