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
@dataclass(frozen=True)
class Term:
    id: float
    wand_upper: float
    Ft: float
    mean_ft: float
    med_ft: float
    min_ft: float
    max_ft: float
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
        for line in tqdm(lines, desc="read qrys"):
            total += 1
            new_query = Query.from_json(line)
            print(new_query)
            if len(new_query.term_ids) <= hyperparams.default_max_qry_len:
                queries.append(new_query)
            else:
                skipped += 1

    print("skipped queries {} out of {}".format(skipped, total))
    return queries


def bucketize(number, bucket_boundaries):
    result = 0
    if number > bucket_boundaries[-1]:
        return len(bucket_boundaries) - 1
    for boundary in bucket_boundaries:
        result += (number > boundary)
    return result


def create_tensors(queries, dev):
    qry = torch.zeros(len(queries), hyperparams.default_max_qry_len,
                      hyperparams.num_term_params, requires_grad=False, device=dev, dtype=torch.long)
    for qidx, q in enumerate(queries):
        for tidx, t in enumerate(q.mapped_terms):
            qry[qidx, tidx, 0] = bucketize(
                t.wand_upper, hyperparams.const_score_buckets)
            qry[qidx, tidx, 1] = bucketize(t.Ft, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 2] = bucketize(
                t.mean_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 3] = bucketize(
                t.med_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 4] = bucketize(
                t.min_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 5] = bucketize(
                t.max_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 6] = bucketize(
                t.num_ft_geq_256, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 7] = bucketize(
                t.num_ft_geq_128, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 8] = bucketize(
                t.num_ft_geq_64, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 9] = bucketize(
                t.num_ft_geq_32, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 10] = bucketize(
                t.num_ft_geq_16, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 11] = bucketize(
                t.num_ft_geq_8, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 12] = bucketize(
                t.num_ft_geq_4, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 13] = bucketize(
                t.num_ft_geq_2, hyperparams.const_Ft_buckets)
    return qry


def create_thresholds(queries, dev):
    thres = torch.zeros(len(queries), 1, requires_grad=False,
                        device=dev, dtype=torch.float)
    for qidx, q in enumerate(queries):
        thres[qidx] = q.wand_thres
    return thres


class InvertedIndexData(Dataset):
    def __init__(self, args, qry_file):
        self.queries = read_queries(qry_file)
        self.tensor_queries = create_tensors(self.queries, args.device)
        self.tensor_thres = create_thresholds(self.queries, args.device)

        my_print("dataset statistics:", qry_file)
        my_print("\tqueries =", len(self.queries))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.tensor_queries[idx, :], self.tensor_thres[idx, :]
