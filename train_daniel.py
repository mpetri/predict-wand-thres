# coding: utf-8
import argparse
import time
import math
#import torch
#import torch.nn as nn
import numpy as np
#from torch.autograd import Variable
#import torch.optim
from scipy import stats
import pandas as pd
from tqdm import tqdm
from tqdm import trange
#from torch.utils.data import Dataset, DataLoader
from util import my_print, init_log, create_file_name
import hyperparams
import data_loader
import models
import os
from typing import List

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import pearsonr


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


def query_to_np(query):
    qry_np = np.zeros(hyperparams.default_max_qry_len*hyperparams.num_term_params)
    for idx,t in enumerate(query.term_data):
        qry_np[idx*hyperparams.num_term_params+0] = t.wand_upper
        qry_np[idx*hyperparams.num_term_params+1] = t.q_weight
        qry_np[idx*hyperparams.num_term_params+2] = t.Ft
        qry_np[idx*hyperparams.num_term_params+3] = t.mean_ft
        qry_np[idx*hyperparams.num_term_params+4] = t.med_ft
        qry_np[idx*hyperparams.num_term_params+5] = t.min_ft
        qry_np[idx*hyperparams.num_term_params+6] = t.max_ft
        qry_np[idx*hyperparams.num_term_params+7] = t.mean_doclen
        qry_np[idx*hyperparams.num_term_params+8] = t.med_doclen
        qry_np[idx*hyperparams.num_term_params+9] = t.min_doclen
        qry_np[idx*hyperparams.num_term_params+10] = t.max_doclen
        qry_np[idx*hyperparams.num_term_params+11] = t.num_ft_geq_256
        qry_np[idx*hyperparams.num_term_params+12] = t.num_ft_geq_128
        qry_np[idx*hyperparams.num_term_params+13] = t.num_ft_geq_64
        qry_np[idx*hyperparams.num_term_params+14] = t.num_ft_geq_32
        qry_np[idx*hyperparams.num_term_params+15] = t.num_ft_geq_16
        qry_np[idx*hyperparams.num_term_params+16] = t.num_ft_geq_8
        qry_np[idx*hyperparams.num_term_params+17] = t.num_ft_geq_4
        qry_np[idx*hyperparams.num_term_params+18] = t.num_ft_geq_2
    return qry_np

def read_queries_and_thres(query_file, data_size=5000):
    ### read query file ###
    queries = []
    thres = []
    skipped = 0
    total = 0
    with open(query_file) as fp:
        lines = fp.readlines()
        for line in tqdm(lines, desc="read qrys", unit="qrys"):
            total += 1
            new_query = Query.from_json(line)
            if len(new_query.term_ids) <= hyperparams.default_max_qry_len:
                q_np = query_to_np(new_query)
                queries.append(q_np)
                thres.append(new_query.wand_thres)
                if len(thres) > data_size:
                    break
            else:
                skipped += 1

    print("skipped queries {} out of {}".format(skipped, total))
    return queries, thres


def train_model(X, y):
    model = BayesianRidge()
    model.fit(X, y)
    return model


def get_preds(model, X_test):
    return model.predict(X_test, return_std=True)


def get_reject(mean, std, threshold):
    """
    This returns a vector containing decisions
    based on a reject option:
    - Returns the mean if std < threshold
    - Returns 0 otherwise

    threshold can be a vector
    """
    decisions = std < threshold
    # little hack: cast bools into 0 and 1
    return mean * decisions
                 

def get_asymmetric(mean, std, weight):
    """
    This returns a vector containing decisions
    based on plugging an asymmetric linear loss.
    This collapses to quantiles of the distribution,
    using the formula (w / w+1).
    The weight should tell you how bad is an
    *underestimate*. Examples:
    - if an underestimate is 3 times worse then
      weight=3
    - if an overestimate is 3 times worse then
      weight=1/3

    weight can be a vector, meaning that the
    loss is instance dependent
    """
    pass

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
    parser.add_argument('--data_dir', type=str, required=True, help='query data dir')
    parser.add_argument('--debug',default=False, dest='debug', action='store_true')
    args = parser.parse_args()

    train_file = args.data_dir + "/train.json"
    if args.debug == True:
        train_file = args.data_dir + "/debug.json"
    dev_file = args.data_dir + "/dev.json"
    test_file = args.data_dir + "/test.json"
    
    #queries, thres = read_queries_and_thres(train_file,
    #                                        data_size=1000000)
    queries = np.loadtxt(args.data_dir + "/train_queries.csv")
    thres = np.loadtxt(args.data_dir + "/train_thres.csv")

    #for q,t in zip(queries,thres):
    #    print("q {} thres {}".format(q,t))

    test_queries, test_thres = read_queries_and_thres(test_file)
        
    #for q,t in zip(test_queries,test_thres):
    #    print("q {} thres {}".format(q,t))

    model = train_model(queries, thres)
    predictions = get_preds(model, test_queries)
    mean_preds, std_preds = predictions
    #mean_preds = np.exp(mean_preds)
    print(MAE(mean_preds, test_thres))
    print(np.sqrt(MSE(mean_preds, test_thres)))
    print(pearsonr(mean_preds, test_thres))
    import ipdb; ipdb.set_trace()
