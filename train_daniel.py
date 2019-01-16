# coding: utf-8
import argparse
import time
import math
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
from tqdm import trange
#from torch.utils.data import Dataset, DataLoader
from util import my_print, init_log, create_file_name
import hyperparams
import data_loader
import os
from typing import List

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.stats import pearsonr


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

    test_queries, test_thres = data_loader.read_queries_and_thres(test_file)
        
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
