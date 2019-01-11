import pickle as pickle
import argparse
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

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

from data_loader import Term


def read_terms(terms_file):
    statinfo = os.stat(terms_file)
    file_size = statinfo.st_size
    prev_pos = 0
    with tqdm(total=file_size, unit='T', desc='read_terms', unit_scale=True, unit_divisor=1000) as pbar:
        with open(file_name, encoding='utf-8', newline='\n', errors='ignore') as f:
            line = f.readline()
            while line:
                terms.append(Term.from_json(line))
                pbar.update(f.tell() - prev_pos)
                prev_pos = f.tell()
                line = f.readline()
    pbar.close()
    return terms


parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--terms', type=str, required=True, help='term data')
parser.add_argument('--out', type=str, required=True, help='output file')
args = parser.parse_args()

terms = read_terms(args.terms)

with open(args.out, "wb") as f:
    pickle.dump(terms, f)
