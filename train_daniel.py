# coding: utf-8
import argparse
import time
import math
import numpy as np
from scipy import stats
#import pandas as pd
from tqdm import tqdm
from tqdm import trange
#from torch.utils.data import Dataset, DataLoader
from util import my_print, init_log, create_file_name
import hyperparams
from data_loader import read_queries_and_thres
import os
from typing import List
import joblib
import json

from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import pearsonr


def train_model(X, y):
    model = BayesianRidge()
    #model = ARDRegression()
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
    dist = stats.norm(mean, std)
    quantile = weight / (weight + 1)
    return dist.ppf(quantile)

def save_test_preds(predictions, ids, term_ids):
    weights = [2,3,4,5,9,19,99]
    to_dump = []
    means, stds = predictions
    for mean, std, _id, t_id in zip(means, stds, ids, term_ids):
        print(_id)
        print(t_id)
        query = {}
        query['id'] = _id
        query['term_ids'] = t_id
        query['mean'] = mean
        query['std'] = std
        for w in weights:
            q = '%.2f' % (w / (w+1))
            val = get_asymmetric(mean, std, w)
            query[q] = val
            inv_q = '%.2f' % ((1/w) / ((1/w)+1))
            val = get_asymmetric(mean, std, (1/w))
            query[inv_q] = val
        to_dump.append(query)
    with open('test_preds.json', 'w') as f:
        for query in to_dump:
            f.write(json.dumps(query) + '\n')
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
    parser.add_argument('--data_dir', type=str, required=True, help='query data dir')
    parser.add_argument('--debug',default=False, dest='debug', action='store_true')
    parser.add_argument('--model', default=False, action='store_true')
    args = parser.parse_args()

    train_file = args.data_dir + "/train.json"
    if args.debug == True:
        train_file = args.data_dir + "/debug.json"
    dev_file = args.data_dir + "/dev.json"
    test_file = args.data_dir + "/test.json"

    print(train_file)

    if not args.model:
        queries, thres, _, _ = read_queries_and_thres(train_file, data_size=40000000)
        print("finish read")
        model = train_model(queries, thres)
        print("finish model")
        joblib.dump(model, "model.joblib")
    else:
        model = joblib.load("model.joblib")

    test_queries, test_thres10, test_thres100, test_thres1000, test_ids, test_term_ids  = read_queries_and_thres(test_file)
   
    predictions = get_preds(model, test_queries)
    mean_preds, std_preds = predictions
    #mean_preds = np.exp(mean_preds)
    print(MAE(mean_preds, test_thres10))
    print(MAE(mean_preds, test_thres100))
    print(MAE(mean_preds, test_thres1000))
    print(np.sqrt(MSE(mean_preds, test_thres10)))
    print(np.sqrt(MSE(mean_preds, test_thres100)))
    print(np.sqrt(MSE(mean_preds, test_thres1000)))
    print(pearsonr(mean_preds, test_thres10))
    print(pearsonr(mean_preds, test_thres100))
    print(pearsonr(mean_preds, test_thres1000))

    save_test_preds(predictions, test_ids, test_term_ids)
    import ipdb; ipdb.set_trace()
