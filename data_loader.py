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

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

@dataclass(frozen=False)
class Term:
    id: float = 0
    q_weight: float = 0.0
    Ft: float = 0.0
    mean_ft: float = 0.0
    med_ft: float = 0.0
    min_ft: float = 0.0
    max_ft: float = 0.0
    mean_doclen: float = 0.0
    med_doclen: float = 0.0
    min_doclen: float = 0.0
    max_doclen: float = 0.0
    num_ft_geq_256: float = 0.0
    num_ft_geq_128: float = 0.0
    num_ft_geq_64: float = 0.0
    num_ft_geq_32: float = 0.0
    num_ft_geq_16: float = 0.0
    num_ft_geq_8: float = 0.0
    num_ft_geq_4: float = 0.0
    num_ft_geq_2: float = 0.0
    block_score_1: float = 0.0
    block_score_2: float = 0.0
    block_score_4: float = 0.0
    block_score_8: float = 0.0
    block_score_16: float = 0.0
    block_score_32: float = 0.0
    block_score_64: float = 0.0
    block_score_128: float = 0.0
    block_score_256: float = 0.0
    block_score_512: float = 0.0
    block_score_1024: float = 0.0
    block_score_2048: float = 0.0
    block_score_4096: float = 0.0
    block_score_small: float = 0.0

@dataclass(frozen=False)
class Query:
    id: int = 0
    wand_thres_10: float = 0.0
    wand_thres_100: float = 0.0
    wand_thres_1000: float = 0.0
    term_ids: List[float] = field(default_factory=list)
    term_data: List[Term] = field(default_factory=list)

def query_to_np(query):
    qry_np = np.zeros(hyperparams.default_max_qry_len*hyperparams.num_term_params)
    for idx,t in enumerate(query.term_data):

        qry_np[idx*hyperparams.num_term_params+0] = t.q_weight
        qry_np[idx*hyperparams.num_term_params+1] = t.Ft

        qry_np[idx*hyperparams.num_term_params+2] = t.mean_ft
        qry_np[idx*hyperparams.num_term_params+3] = t.med_ft
        qry_np[idx*hyperparams.num_term_params+4] = t.min_ft
        qry_np[idx*hyperparams.num_term_params+5] = t.max_ft

        qry_np[idx*hyperparams.num_term_params+6] = t.mean_doclen
        qry_np[idx*hyperparams.num_term_params+7] = t.med_doclen
        qry_np[idx*hyperparams.num_term_params+8] = t.min_doclen
        qry_np[idx*hyperparams.num_term_params+9] = t.max_doclen

        qry_np[idx*hyperparams.num_term_params+10] = t.num_ft_geq_256
        qry_np[idx*hyperparams.num_term_params+11] = t.num_ft_geq_128
        qry_np[idx*hyperparams.num_term_params+12] = t.num_ft_geq_64
        qry_np[idx*hyperparams.num_term_params+13] = t.num_ft_geq_32
        qry_np[idx*hyperparams.num_term_params+14] = t.num_ft_geq_16
        qry_np[idx*hyperparams.num_term_params+15] = t.num_ft_geq_8
        qry_np[idx*hyperparams.num_term_params+16] = t.num_ft_geq_4
        qry_np[idx*hyperparams.num_term_params+17] = t.num_ft_geq_2

        qry_np[idx*hyperparams.num_term_params+18] = t.block_score_1
        qry_np[idx*hyperparams.num_term_params+19] = t.block_score_2
        qry_np[idx*hyperparams.num_term_params+20] = t.block_score_4
        qry_np[idx*hyperparams.num_term_params+21] = t.block_score_8
        qry_np[idx*hyperparams.num_term_params+22] = t.block_score_16
        qry_np[idx*hyperparams.num_term_params+23] = t.block_score_32
        qry_np[idx*hyperparams.num_term_params+24] = t.block_score_64
        qry_np[idx*hyperparams.num_term_params+25] = t.block_score_128
        qry_np[idx*hyperparams.num_term_params+26] = t.block_score_256
        qry_np[idx*hyperparams.num_term_params+27] = t.block_score_512
        qry_np[idx*hyperparams.num_term_params+28] = t.block_score_1024
        qry_np[idx*hyperparams.num_term_params+29] = t.block_score_2048
        qry_np[idx*hyperparams.num_term_params+30] = t.block_score_4096
        qry_np[idx*hyperparams.num_term_params+31] = t.block_score_small

    return qry_np

def read_queries_and_thres(query_file, data_size=5000):
    ### read query file ###
    queries = []
    thres_10 = []
    thres_100 = []
    thres_1000 = []
    query_ids = []
    query_term_ids = []
    skipped = 0
    total = 0
    with open(query_file) as fp:
        lines = fp.readlines()
        for line in tqdm(lines, desc="read qrys", unit="qrys"):
            total += 1
            qry_dict = rapidjson.loads(line)
            new_query = Query()
            new_query.term_ids = qry_dict["term_ids"]
            if len(new_query.term_ids) <= hyperparams.default_max_qry_len:
                new_query.id = qry_dict["id"]
                new_query.wand_thres_10 = float(qry_dict["wand_thres_10"])
                new_query.wand_thres_100 = float(qry_dict["wand_thres_100"])
                new_query.wand_thres_1000 = float(qry_dict["wand_thres_1000"])
                for t in qry_dict["term_data"]:
                    new_term = Term()
                    new_term.id = t["id"]
                    new_term.q_weight = float(t["q_weight"])
                    new_term.Ft = float(t["Ft"])
                    new_term.mean_ft = float(t["mean_ft"])
                    new_term.med_ft = float(t["med_ft"])
                    new_term.min_ft = float(t["min_ft"])
                    new_term.max_ft = float(t["max_ft"])
                    new_term.mean_doclen = float(t["mean_doclen"])
                    new_term.med_doclen = float(t["med_doclen"])
                    new_term.min_doclen = float(t["min_doclen"])
                    new_term.max_doclen = float(t["max_doclen"])

                    new_term.num_ft_geq_256 = float(t["num_ft_geq_256"])
                    new_term.num_ft_geq_128 = float(t["num_ft_geq_128"])
                    new_term.num_ft_geq_64 = float(t["num_ft_geq_64"])
                    new_term.num_ft_geq_32 = float(t["num_ft_geq_32"])
                    new_term.num_ft_geq_16 = float(t["num_ft_geq_16"])
                    new_term.num_ft_geq_8 = float(t["num_ft_geq_8"])
                    new_term.num_ft_geq_4 = float(t["num_ft_geq_4"])
                    new_term.num_ft_geq_2 = float(t["num_ft_geq_2"])

                    new_term.block_score_1 = float(t["block_score_1"])
                    new_term.block_score_2 = float(t["block_score_2"])
                    new_term.block_score_4 = float(t["block_score_4"])
                    new_term.block_score_8 = float(t["block_score_8"])
                    new_term.block_score_16 = float(t["block_score_16"])
                    new_term.block_score_32 = float(t["block_score_32"])
                    new_term.block_score_64 = float(t["block_score_64"])
                    new_term.block_score_128 = float(t["block_score_128"])
                    new_term.block_score_256 = float(t["block_score_256"])
                    new_term.block_score_512 = float(t["block_score_512"])
                    new_term.block_score_1024 = float(t["block_score_1024"])
                    new_term.block_score_2048 = float(t["block_score_2048"])
                    new_term.block_score_4096 = float(t["block_score_4096"])
                    new_term.block_score_small = float(t["block_score_small"])

                    new_query.term_data.append(new_term)

                def sort_Ft(val): 
                    return val.Ft
                new_query.term_data.sort(key = sort_Ft)  
                q_np = query_to_np(new_query)
                queries.append(q_np)
                thres_10.append(new_query.wand_thres_10)
                thres_100.append(new_query.wand_thres_100)
                thres_1000.append(new_query.wand_thres_1000)
                query_ids.append(new_query.id)
                query_term_ids.append(new_query.term_ids)
                if data_size != 0 and len(thres_10) > data_size:
                    break
            else:
                skipped += 1

    print("skipped queries {} out of {}".format(skipped, total))
    return queries, thres_10, thres_100, thres_1000, query_ids, query_term_ids


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
    for qidx, q in enumerate(tqdm(queries, desc="bucketize qrys", unit="qrys")):
        for tidx, t in enumerate(q.term_data):
            qry[qidx, tidx, 0] = bucketize(
                t.wand_upper, hyperparams.const_score_buckets)
            qry[qidx, tidx, 1] = bucketize(
                t.q_weight, hyperparams.const_score_buckets)
                
            qry[qidx, tidx, 2] = bucketize(t.Ft, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 3] = bucketize(
                t.mean_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 4] = bucketize(
                t.med_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 5] = bucketize(
                t.min_ft, hyperparams.const_freq_buckets)
            qry[qidx, tidx, 6] = bucketize(
                t.max_ft, hyperparams.const_freq_buckets)

            qry[qidx, tidx, 7] = bucketize(
                t.mean_ft, hyperparams.const_doc_len_buckets)
            qry[qidx, tidx, 8] = bucketize(
                t.med_ft, hyperparams.const_doc_len_buckets)
            qry[qidx, tidx, 9] = bucketize(
                t.min_ft, hyperparams.const_doc_len_buckets)
            qry[qidx, tidx, 10] = bucketize(
                t.max_ft, hyperparams.const_doc_len_buckets)

            qry[qidx, tidx, 11] = bucketize(
                t.num_ft_geq_256, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 12] = bucketize(
                t.num_ft_geq_128, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 13] = bucketize(
                t.num_ft_geq_64, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 14] = bucketize(
                t.num_ft_geq_32, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 15] = bucketize(
                t.num_ft_geq_16, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 16] = bucketize(
                t.num_ft_geq_8, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 17] = bucketize(
                t.num_ft_geq_4, hyperparams.const_Ft_buckets)
            qry[qidx, tidx, 18] = bucketize(
                t.num_ft_geq_2, hyperparams.const_Ft_buckets)
    return qry

def create_tensors_from_np(queries, dev):
    qry = torch.zeros(len(queries), hyperparams.default_max_qry_len * hyperparams.num_term_params, requires_grad=False, device=dev, dtype=torch.float)
    for idx,q in enumerate(queries):
        qry[idx,:] = torch.as_tensor(q)
    return qry

def create_thresholds(thres_lst, dev):
    thres = torch.zeros(len(thres_lst), 1, requires_grad=False,
                        device=dev, dtype=torch.float)
    for qidx, t in enumerate(thres_lst):
        thres[qidx] = t
    return thres


class InvertedIndexData(Dataset):
    def __init__(self, args, qry_file):
        self.queries,self.thres_10,self.thres_100,self.thres_1000 = read_queries_and_thres(qry_file,0)
        self.tensor_queries = create_tensors_from_np(self.queries, args.device)
        self.tensor_thres = create_thresholds(self.thres_10, args.device)

        my_print("dataset statistics:", qry_file)
        my_print("\tqueries =", len(self.queries))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.tensor_queries[idx, :], self.tensor_thres[idx, :]
