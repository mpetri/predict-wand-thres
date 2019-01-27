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
    k10_max: float = 0.0
    k100_max: float = 0.0
    k1000_max: float = 0.0


@dataclass(frozen=False)
class Query:
    id: int = 0
    wand_thres_10: float = 0.0
    wand_thres_100: float = 0.0
    wand_thres_1000: float = 0.0
    qmax_k_10: float = 0.0
    qmax_k_100: float = 0.0
    qmax_k_1000: float = 0.0
    term_ids: List[float] = field(default_factory=list)
    term_data: List[Term] = field(default_factory=list)


def query_to_np(query):
    qry_np = np.zeros(3 + hyperparams.default_max_qry_len *
                      hyperparams.num_term_params)
    qry_np[0] = query.qmax_k_10
    qry_np[1] = query.qmax_k_100
    qry_np[2] = query.qmax_k_1000
    for idx, t in enumerate(query.term_data):

        qry_np[3 + (idx * hyperparams.num_term_params + 0)] = t.q_weight
        qry_np[3 + (idx * hyperparams.num_term_params + 1)] = t.Ft

        qry_np[3 + (idx * hyperparams.num_term_params + 2)] = t.k10_max
        qry_np[3 + (idx * hyperparams.num_term_params + 3)] = t.k100_max
        qry_np[3 + (idx * hyperparams.num_term_params + 4)] = t.k1000_max

        qry_np[3 + idx * hyperparams.num_term_params + 5] = t.mean_ft
        qry_np[3 + idx * hyperparams.num_term_params + 6] = t.med_ft
        qry_np[3 + idx * hyperparams.num_term_params + 7] = t.min_ft
        qry_np[3 + idx * hyperparams.num_term_params + 8] = t.max_ft

        qry_np[3 + idx * hyperparams.num_term_params + 9] = t.mean_doclen
        qry_np[3 + idx * hyperparams.num_term_params + 10] = t.med_doclen
        qry_np[3 + idx * hyperparams.num_term_params + 11] = t.min_doclen
        qry_np[3 + idx * hyperparams.num_term_params + 12] = t.max_doclen

        qry_np[3 + idx * hyperparams.num_term_params + 13] = t.num_ft_geq_256
        qry_np[3 + idx * hyperparams.num_term_params + 14] = t.num_ft_geq_128
        qry_np[3 + idx * hyperparams.num_term_params + 15] = t.num_ft_geq_64
        qry_np[3 + idx * hyperparams.num_term_params + 16] = t.num_ft_geq_32
        qry_np[3 + idx * hyperparams.num_term_params + 17] = t.num_ft_geq_16
        qry_np[3 + idx * hyperparams.num_term_params + 18] = t.num_ft_geq_8
        qry_np[3 + idx * hyperparams.num_term_params + 19] = t.num_ft_geq_4
        qry_np[3 + idx * hyperparams.num_term_params + 20] = t.num_ft_geq_2

        qry_np[3 + idx * hyperparams.num_term_params + 21] = t.block_score_1
        qry_np[3 + idx * hyperparams.num_term_params + 22] = t.block_score_2
        qry_np[3 + idx * hyperparams.num_term_params + 23] = t.block_score_4
        qry_np[3 + idx * hyperparams.num_term_params + 24] = t.block_score_8
        qry_np[3 + idx * hyperparams.num_term_params + 25] = t.block_score_16
        qry_np[3 + idx * hyperparams.num_term_params + 26] = t.block_score_32
        qry_np[3 + idx * hyperparams.num_term_params + 27] = t.block_score_64
        qry_np[3 + idx * hyperparams.num_term_params + 28] = t.block_score_128
        qry_np[3 + idx * hyperparams.num_term_params + 29] = t.block_score_256
        qry_np[3 + idx * hyperparams.num_term_params + 30] = t.block_score_512
        qry_np[3 + idx * hyperparams.num_term_params + 31] = t.block_score_1024
        qry_np[3 + idx * hyperparams.num_term_params + 32] = t.block_score_2048
        qry_np[3 + idx * hyperparams.num_term_params + 33] = t.block_score_4096
        qry_np[3 + idx * hyperparams.num_term_params + 34] = t.block_score_small

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
                new_query.qmax_k_10 = float(qry_dict["max_qk10"])
                new_query.qmax_k_100 = float(qry_dict["max_qk100"])
                new_query.qmax_k_1000 = float(qry_dict["max_qk1000"])
                # qid,qlen,k,model,rho,pred,actual
                for t in qry_dict["term_data"]:
                    new_term = Term()
                    new_term.id = t["id"]
                    new_term.q_weight = float(t["q_weight"])
                    new_term.Ft = float(t["Ft"])

                    new_term.k10_max = float(t["k10m"])
                    new_term.k100_max = float(t["k100m"])
                    new_term.k1000_max = float(t["k1000m"])

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
                new_query.term_data.sort(key=sort_Ft)
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


def create_tensors_from_np(queries):
    qry = torch.zeros(len(queries), 3 + hyperparams.default_max_qry_len *
                      hyperparams.num_term_params, requires_grad=False, dtype=torch.float)
    for idx, q in enumerate(queries):
        qry[idx, :] = torch.as_tensor(q)
    return qry


def create_thresholds(thres_lst):
    thres = torch.zeros(len(thres_lst), 1, requires_grad=False,
                        dtype=torch.float)
    for qidx, t in enumerate(thres_lst):
        thres[qidx] = t
    return thres


class InvertedIndexData(Dataset):
    def __init__(self, args, qry_file):
        self.queries, self.thres_10, self.thres_100, self.thres_1000, self.qids, self.qterms = read_queries_and_thres(
            qry_file, 0)
        self.tensor_queries = create_tensors_from_np(self.queries, args.device)
        self.qlens = []
        for qt in self.qterms:
            self.qlens.append(len(qt))

        if args.k == 10:
            self.tensor_thres = create_thresholds(self.thres_10, args.device)
        if args.k == 100:
            self.tensor_thres = create_thresholds(self.thres_100, args.device)
        if args.k == 1000:
            self.tensor_thres = create_thresholds(self.thres_1000, args.device)

        my_print("dataset statistics:", qry_file)
        my_print("\tqueries =", len(self.queries))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.tensor_queries[idx, :], self.tensor_thres[idx, :]
