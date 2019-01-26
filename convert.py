import pickle as pickle
import os
from tqdm import tqdm
from tqdm import trange
import io
import numpy as np
import sys
import array
from util import my_print
import collections
import hyperparams
import rapidjson


import argparse

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--qry', type=str, required=True, help='query file')
parser.add_argument('--fix', type=str, required=True, help='fix file')
args = parser.parse_args()

def read_queries(query_file,fix_file):
    fix_k10m = {}
    fix_k100m = {}
    fix_k1000m = {}
    with open(fix_file) as fix_fp:
        fix_lines = fix_fp.readlines()
        for line in tqdm(fix_lines):
            splits = line.split(",")
            term_id = int(splits[0])
            k10m = float(splits[1])
            k100m = float(splits[2])
            k1000m = float(splits[3])
            fix_k10m[term_id] = k10m
            fix_k100m[term_id] = k100m
            fix_k1000m[term_id] = k1000m
    with open(query_file) as fp:
        lines = fp.readlines()
        for line in tqdm(lines, desc="read qrys", unit="qrys"):
            qry_dict = rapidjson.loads(line)
            max_qk10 = 0
            max_qk100 = 0
            max_qk1000 = 0
            for t in qry_dict["term_data"]:
                new_term_id = t["id"]
                t["k10m"] = fix_k10m[new_term_id]
                t["k100m"] = fix_k100m[new_term_id]
                t["k1000m"] = fix_k1000m[new_term_id]
                max_qk10 = max(max_qk10,fix_k10m[new_term_id])
                max_qk100 = max(max_qk100,fix_k100m[new_term_id])
                max_qk1000 = max(max_qk1000,fix_k1000m[new_term_id])
            qry_dict["max_qk10"] = max_qk10
            qry_dict["max_qk100"] = max_qk100
            qry_dict["max_qk1000"] = max_qk1000
            print(rapidjson.dumps(qry_dict))

read_queries(args.qry,args.fix)

