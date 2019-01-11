import pickle as pickle
import os
from tqdm import tqdm
from tqdm import trange
import io
import numpy as np
import sys
import array
from util import my_print

from dataclasses import dataclass
from dataclasses_json import dataclass_json

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass_json
@dataclass
class Query:
    id: int
    thres: float
    terms: List[int]


@dataclass_json
@dataclass
class Term:
    id: int
    Ft: float
    Meanft: float
    Medft: float
    Minft: float
    Maxft: float
    Largerft256: float
    Largerft128: float
    Largerft64: float
    Largerft32: float
    Largerft16: float
    Largerft8: float
    Largerft4: float
    Largerft2: float


def read_queries(query_file):
    ### read query file ###
    queries = []
    with open(query_file) as fp:
        for line in fp:
            queries.append(Query.from_json(line))
    return queries


def read_terms(terms_file):
    ### read terms file ###
    with open(terms_file) as fp:
        for line in fp:
            thres.append(Threshold.from_json(line))


class InvertedIndexData(Dataset):
    def __init__(self, args):
        self.queries = read_queries(args.queries)
        self.terms = read_terms(args.terms)

        my_print("dataset statistics:")
        my_print("\tqueries =", len(self.queries))
        my_print("\tterms =", len(self.terms))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.thresholds[idx]
