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
    terms = []
    with open(terms_file, encoding='utf-8', newline='\n', errors='ignore') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="read_terms", unit_divisor=1000, unit='T'):

            terms.append(Term.from_json(line))
    return terms


parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--terms', type=str, required=True, help='term data')
parser.add_argument('--out', type=str, required=True, help='output file')
args = parser.parse_args()

terms = read_terms(args.terms)

with open(args.out, "wb") as f:
    pickle.dump(terms, f)
