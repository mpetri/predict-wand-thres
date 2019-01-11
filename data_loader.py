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

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


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

@dataclass_json
@dataclass
class Query:
    id: int
    thres: float
    terms: List[int]
    mapped_terms: List[Term] = field(default_factory=list)

def read_queries(query_file,terms):
    ### read query file ###
    queries = []
    skipped = 0
    total = 0
    with open(query_file) as fp:
        for line in fp:
            total += 1
            new_query = Query.from_json(line)
            if len(new_query.terms) <= hyperparams.default_max_qry_len:
                queries.append(Query.from_json(line))
            else:
                skipped += 1

    for q in queries:
        for t in q.terms:
            if t < len(terms):
                 q.mapped_terms.append(terms[t])

    print("skipped queries {} out of {}".format(skipped,total))
    return queries

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Terms':
            from data_loader import Term
            return Term
        return super().find_class(module, name)

def read_terms(terms_file):
    ### read terms file ###
    terms = []
    with open(terms_file,"rb") as f:
        terms = CustomUnpickler(f).load()
    return terms

def bucketize(number,bucket_boundaries):
    result = 0
    if number > bucket_boundaries[-1]:
        return len(bucket_boundaries)-1
    for boundary in bucket_boundaries:
        result += (number > boundary)
    return result

def create_tensors(queries,dev):
    qry = torch.zeros(len(queries),hyperparams.default_max_qry_len,hyperparams.num_term_params,requires_grad=False,device=dev,dtype=torch.long)
    print(qry)
    for qidx,q in enumerate(queries):
        for tidx,t in enumerate(q.mapped_terms):
            qry[qidx,tidx,0] = bucketize(t.Ft,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,1] = bucketize(t.Meanft,hyperparams.const_freq_buckets)
            qry[qidx,tidx,2] = bucketize(t.Medft,hyperparams.const_freq_buckets)
            qry[qidx,tidx,3] = bucketize(t.Minft,hyperparams.const_freq_buckets)
            qry[qidx,tidx,4] = bucketize(t.Maxft,hyperparams.const_freq_buckets)
            qry[qidx,tidx,5] = bucketize(t.Largerft256,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,6] = bucketize(t.Largerft128,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,7] = bucketize(t.Largerft64,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,8] = bucketize(t.Largerft32,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,9] = bucketize(t.Largerft16,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,10] = bucketize(t.Largerft8,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,11] = bucketize(t.Largerft4,hyperparams.const_Ft_buckets)
            qry[qidx,tidx,12] = bucketize(t.Largerft2,hyperparams.const_Ft_buckets)

    return qry

def create_thresholds(queries,dev):
    thres = torch.zeros(len(queries),1,requires_grad=False,device=dev,dtype=torch.float)
    print(thres)
    for qidx,q in enumerate(queries):
        thres[qidx] = q.thres
    return thres

class InvertedIndexData(Dataset):
    def __init__(self, args,qry_file):
        self.terms = read_terms(args.terms)
        self.queries = read_queries(qry_file,self.terms)
        self.tensor_queries = create_tensors(self.queries,args.device)
        self.tensor_thres = create_thresholds(self.queries,args.device)

        my_print("dataset statistics:",qry_file)
        my_print("\tqueries =", len(self.queries))
        my_print("\tterms =", len(self.terms))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.tensor_queries[idx,:],self.tensor_thres[idx,:]
