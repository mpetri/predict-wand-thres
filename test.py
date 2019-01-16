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

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--queries', type=str, required=True, help='query data')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--device', default="cpu", type=str,
                    required=False, help='compute device')
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

with torch.no_grad():
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=args.device)
        print(model)
        model.eval()
        print("id;predicted;actual;time_ms")
        total_time_ms = 0
        for qry, thres in dataset:
            qry = qry.view(1, qry.size(0))
            start = time.time()
            pred_thres = model(qry.to(args.device))
            elapsed = time.time() - start
            total_time_ms += elapsed * 1000
            print("{};{};{};{}".format(qry.id,pred_thres.item(),
                                    thres.item(), elapsed * 1000))
        print("mean time per qry {}".format(
            float(total_time_ms) / float(len(dataset))), file=sys.stderr)
