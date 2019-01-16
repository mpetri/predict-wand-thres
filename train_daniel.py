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
from typing import List

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--data_dir', type=str, required=True, help='query data dir')
args = parser.parse_args()

train_file = args.data_dir + "/train.json"
dev_file = args.data_dir + "/dev.json"
test_file = args.data_dir + "/test.json"

@dataclass_json
@dataclass(frozen=True)
class Term:
    id: float
    wand_upper: float
    q_weight: float
    Ft: float
    mean_ft: float
    med_ft: float
    min_ft: float
    max_ft: float
    mean_doclen: float
    med_doclen: float
    min_doclen: float
    max_doclen: float
    num_ft_geq_256: float
    num_ft_geq_128: float
    num_ft_geq_64: float
    num_ft_geq_32: float
    num_ft_geq_16: float
    num_ft_geq_8: float
    num_ft_geq_4: float
    num_ft_geq_2: float


@dataclass_json
@dataclass(frozen=True)
class Query:
    id: int
    wand_thres: float
    term_ids: List[float]
    term_data: List[Term] = field(default_factory=list)


def read_queries(query_file):
    ### read query file ###
    queries = []
    skipped = 0
    total = 0
    with open(query_file) as fp:
        lines = fp.readlines()
        for line in tqdm(lines, desc="read qrys", unit="qrys"):
            total += 1
            new_query = Query.from_json(line)
            if len(new_query.term_ids) <= hyperparams.default_max_qry_len:
                queries.append(new_query)
            else:
                skipped += 1

    print("skipped queries {} out of {}".format(skipped, total))
    return queries

queries = read_queries(train_file)

for q in queries:
    print(q)


