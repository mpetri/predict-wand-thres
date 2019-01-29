# coding: utf-8
import argparse
import time
import math
import numpy as np
from scipy import stats
from tqdm import tqdm
from tqdm import trange
from util import my_print, init_log, create_file_name
import hyperparams
from data_loader import read_queries_and_thres
import os
from typing import List
import joblib
import json
import timeit

from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import sklearn.preprocessing as pp
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
                 

def get_asymmetric(mean, std, quantile):
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
    #quantile = weight / (weight + 1)
    return dist.ppf(quantile)


def save_test_preds(predictions, ids, term_ids, filename, test_gross, test_thres1000):
    #weights = [2,3,4,5,9,19,99]
    quantiles = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    to_dump = []
    means, stds = predictions
    for mean, std, _id, t_id, gross, test in zip(means, stds, ids, term_ids, test_gross, test_thres1000):
        query = {}
        query['id'] = _id
        query['term_ids'] = t_id
        query['mean'] = mean
        query['std'] = std
        query['gross'] = gross
        query['test1000'] = test
        #for w in weights:
        dist = stats.norm(mean, std)
        for q in quantiles:
            #q = '%.2f' % (w / (w+1))
            #val = get_asymmetric(mean, std, q)
            val = dist.ppf(q)
            query[q] = val
            #inv_q = '%.2f' % ((1/w) / ((1/w)+1))
            #val = get_asymmetric(mean, std, (1/w))
            #query[inv_q] = val
        to_dump.append(query)
    with open(filename, 'w') as f:
        for query in to_dump:
            f.write(json.dumps(query) + '\n')

            
def time_pred(model, query):
    mean, std = model.predict(query, return_std=True)
    dist = stats.norm(mean, std)
    #q50 = 1 / (1+1)
    return dist#, dist.ppf(q50)


def time_quantile(dist, q):
    return dist.ppf(q)


def print_metrics(predictions, test_thres, QS):
    means, stds = predictions
    quantiles = []
    mups = []
    overs = []
    for mean, std, test in zip(means, stds, test_thres):
        dist = stats.norm(mean, std)
        quantile = []
        for q in QS:
            quantile.append(dist.ppf(q))
        quantiles.append(quantile)
    # Get metrics
    quantiles = np.array(quantiles)
    for i, q in enumerate(QS):
        qs = quantiles[:, i]
        #for j, q in enumerate(qs):
        #    if q < 0:
        #        print("%d, %f" % (j, q))
        vec = (qs >= test_thres)
        not_vec = (qs < test_thres)
        over = np.sum(vec) / float(len(test_thres)+1)
        under = np.compress(not_vec, qs)
        under[under < 0] = 0.0
        under_test = np.compress(not_vec, test_thres)
        mup = np.mean(under / under_test)
        mups.append(mup)
        overs.append(over)
    return mups, overs


def get_mup(preds, test):
    not_vec = (preds < test)
    under = np.compress(not_vec, preds)
    under_test = np.compress(not_vec, test)
    return np.mean(under / under_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
    parser.add_argument('--data_dir', type=str, required=True, help='query data dir')
    parser.add_argument('--debug',default=False, dest='debug', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='model file, creates a new one if does not exist or the --overwrite flag is set, otherwise it loads the model')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('--test', type=str, default=None)
    args = parser.parse_args()

    train_file = args.data_dir + "/train.json"
    if args.debug == True:
        train_file = args.data_dir + "/debug.json"
    dev_file = args.data_dir + "/dev.json"
    if args.test is None:
        test_file = args.data_dir + "/test.json"
    else:
        test_file = args.test

    print(test_file)

    if (not os.path.exists(args.model)) or args.overwrite:
        queries, thres10, thres100, thres1000, ids, term_ids, _ = read_queries_and_thres(train_file, data_size=40000000)
        print("finish read")

        # Standardize
        queries = np.array(queries)
        scaler = pp.StandardScaler(queries.shape[1])
        scaler.fit(queries)
        queries = scaler.transform(queries)
        
        if args.k == 10:
            thres = thres10
        elif args.k == 100:
            thres = thres100
        elif args.k == 1000:
            thres = thres1000
        
        model = train_model(queries, thres)
        print("finish model")
        joblib.dump(model, args.model)
        joblib.dump(scaler, args.model + '.scaler')
    else:
        model = joblib.load(args.model)
        scaler = joblib.load(args.model + '.scaler')

    test_queries, test_thres10, test_thres100, test_thres1000, test_ids, test_term_ids, test_gross  = read_queries_and_thres(test_file, data_size=20000)

    # Standardize
    test_queries = scaler.transform(test_queries)
    print(test_queries.shape)
    #import sys; sys.exit(0)
    test_gross = np.array(test_gross)
    test_gross_10 = test_gross[:, 0]
    test_gross_100 = test_gross[:, 1]
    test_gross_1000 = test_gross[:, 2]
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

    print('*' * 80)
    print("GROSS PREDICTORS")
    print('*' * 80)
    print(MAE(test_gross_10, test_thres10))
    print(MAE(test_gross_100, test_thres100))
    print(MAE(test_gross_1000, test_thres1000))
    print(np.sqrt(MSE(test_gross_10, test_thres10)))
    print(np.sqrt(MSE(test_gross_100, test_thres100)))
    print(np.sqrt(MSE(test_gross_1000, test_thres1000)))
    print(pearsonr(test_gross_10, test_thres10))
    print(pearsonr(test_gross_100, test_thres100))
    print(pearsonr(test_gross_1000, test_thres1000))
    print(get_mup(test_gross_10, test_thres10))
    print(get_mup(test_gross_100, test_thres100))
    print(get_mup(test_gross_1000, test_thres1000))
    
    print('*' * 80)
    print("MUP and over")
    print('*' * 80)
    QS = [0.5, 0.3, 0.1, 0.05, 0.01]
    if args.k == 10:
        test_thres = test_thres10
    elif args.k == 100:
        test_thres = test_thres100
    elif args.k == 1000:
        test_thres = test_thres1000
    print(predictions[0][:10])
    print(predictions[1][:10])
    print(test_thres[:10])
    print(print_metrics(predictions, test_thres, QS))
    
    
    REP = 1000
    toy_query = test_queries[0].reshape(1, -1)
    time = timeit.timeit("time_pred(model, toy_query)", number=REP, setup="from __main__ import time_pred,model,toy_query")
    print("%.5f microsec per query" % ((time/REP) * 1000 * 1000))
    
    dist = time_pred(model, toy_query)
    time = timeit.timeit("time_quantile(dist, 0.5)", number=REP, setup="from __main__ import time_quantile,dist")
    print("%.5f microsec per quantile" % ((time/REP) * 1000 * 1000))

    if args.test is not None:
        filename = args.test + '.k%d' % args.k
        save_test_preds(predictions, test_ids, test_term_ids, filename, test_gross_10, test_thres10)
    import ipdb; ipdb.set_trace()
