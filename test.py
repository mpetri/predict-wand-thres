# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim
from scipy import stats
import pandas as pd
from tqdm import tqdm
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from util import my_print, init_log, create_file_name
import hyperparams
import data_loader
import models
import os
import sys
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--queries', type=str, required=True, help='query data')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--device', default="cpu", type=str,
                    required=False, help='compute device')
parser.add_argument('--k', type=int, required=True, help='prediction depth')
args = parser.parse_args()
torch.set_num_threads(hyperparams.default_threads)
print("Parameters:")
for k, v in sorted(vars(args).items()):
    print("\t{0}: {1}".format(k, v))

# Set the random seed manually for reproducibility.
torch.manual_seed(hyperparams.random_seed)

###############################################################################
# Load data
###############################################################################
if torch.cuda.is_available():
    if args.device != "cpu":
        torch.cuda.set_device(int(args.device.split(":")[1]))
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

else:
    args.device = torch.device('cpu')
print("Using torch device:", args.device)

dataset = data_loader.InvertedIndexData(args, args.queries)
qids = dataset.qids
qlens = dataset.qlens

with torch.no_grad():
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=args.device)
        print(model)
        model.eval()
        total_time_ms = 0
        error_sum = 0.0
        num_over_predicted = 0
        preds = []
        actual = []
        under_preds = []
        min_elapsed = 9999999999
        for qry, thres in dataset:
            qry = qry.view(1, qry.size(0))
            start = time.time()
            pred_thres = model(qry.to(args.device))
            elapsed = time.time() - start
            min_elapsed = min(elapsed,min_elapsed)
            total_time_ms += elapsed * 1000
            if pred_thres.item() - thres.item() > 0:
                num_over_predicted += 1

            pred_act = max(0.0, pred_thres.item())
            act_thres = thres.item()
            preds.append(pred_act)
            actual.append(act_thres)

            if thres.item() >= pred_thres.item():
                under_preds.append(pred_act / thres.item())


        MUE = np.mean(np.asarray(under_preds))
        RHO = pearsonr(actual, preds)
        percent_over = float(num_over_predicted * 100) / float(len(dataset))
        for qid,qlen,pred,act in zip(qids,qlens,preds,actual):
            print("{},{},{},{},{},{},{}".format(qid,qlen,args.k,args.model,RHO[0],pred,act))

