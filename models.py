import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy import stats
import array
import hyperparams
from util import my_print
import torch.nn.functional as F


class Complex(nn.Module):
    def __init__(self, embed_size, num_layers=hyperparams.default_num_layers):
        super(Simple, self).__init__()
        my_print("Initializing Simple model")
        self.WAND_upper = nn.Embedding(
            len(hyperparams.const_score_buckets), embed_size)
        self.q_weight = nn.Embedding(
            len(hyperparams.const_score_buckets), embed_size)

        self.Ft = nn.Embedding(len(hyperparams.const_Ft_buckets), embed_size)
        self.Meanft = nn.Embedding(
            len(hyperparams.const_freq_buckets), embed_size)
        self.Medft = nn.Embedding(
            len(hyperparams.const_freq_buckets), embed_size)
        self.Minft = nn.Embedding(
            len(hyperparams.const_freq_buckets), embed_size)
        self.Maxft = nn.Embedding(
            len(hyperparams.const_freq_buckets), embed_size)

        self.MeanDocLen = nn.Embedding(
            len(hyperparams.const_doc_len_buckets), embed_size)
        self.MedDocLen = nn.Embedding(
            len(hyperparams.const_doc_len_buckets), embed_size)
        self.MinDocLen = nn.Embedding(
            len(hyperparams.const_doc_len_buckets), embed_size)
        self.MaxDocLen = nn.Embedding(
            len(hyperparams.const_doc_len_buckets), embed_size)

        self.Largerft256 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft128 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft64 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft32 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft16 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft8 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft4 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)
        self.Largerft2 = nn.Embedding(
            len(hyperparams.const_Ft_buckets), embed_size)

        self.num_layers = num_layers

        self.input_dim = hyperparams.default_max_qry_len * \
            hyperparams.num_term_params * embed_size
        self.layers = torch.nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module("linear_{}".format(
                i + 1), nn.Linear(self.input_dim, self.input_dim))
            self.layers.add_module("ReLU_{}".format(i + 1), nn.ReLU())
        self.layers.add_module("last_linear", nn.Linear(self.input_dim, 1))

    def forward(self, queries):
        emb_WAND_Upper = self.WAND_upper(queries[:, :, 0])
        emb_q_weight = self.q_weight(queries[:, :, 1])
        emb_Ft = self.Ft(queries[:, :, 2])

        emb_Meanft = self.Meanft(queries[:, :, 3])
        emb_Medft = self.Medft(queries[:, :, 4])
        emb_Minft = self.Minft(queries[:, :, 5])
        emb_Maxft = self.Maxft(queries[:, :, 6])

        emb_MeanDocLen = self.MeanDocLen(queries[:, :, 7])
        emb_MedDocLen = self.MedDocLen(queries[:, :, 8])
        emb_MinDocLen = self.MinDocLen(queries[:, :, 9])
        emb_MaxDocLen = self.MaxDocLen(queries[:, :, 10])

        emb_Largerft256 = self.Largerft256(queries[:, :, 11])
        emb_Largerft128 = self.Largerft128(queries[:, :, 12])
        emb_Largerft64 = self.Largerft64(queries[:, :, 13])
        emb_Largerft32 = self.Largerft32(queries[:, :, 14])
        emb_Largerft16 = self.Largerft16(queries[:, :, 15])
        emb_Largerft8 = self.Largerft8(queries[:, :, 16])
        emb_Largerft4 = self.Largerft4(queries[:, :, 17])
        emb_Largerft2 = self.Largerft2(queries[:, :, 18])

        query_embed = torch.cat((emb_WAND_Upper,emb_q_weight, emb_Ft, emb_Meanft, emb_Medft, emb_Minft, emb_Maxft,emb_MeanDocLen,
         emb_MedDocLen, emb_MinDocLen, emb_MaxDocLen, emb_Largerft256, emb_Largerft128,
                                 emb_Largerft64, emb_Largerft32, emb_Largerft16, emb_Largerft8, emb_Largerft4, emb_Largerft2), 2)
        query_embed = query_embed.view(-1, self.input_dim)
        pred_thres = self.layers(query_embed)
        return pred_thres


class Simple(nn.Module):
    def __init__(self, embed_size, num_layers=hyperparams.default_num_layers):
        super(Simple, self).__init__()
        self.num_layers = num_layers
        self.input_dim = hyperparams.default_max_qry_len * hyperparams.num_term_params
        self.layers = torch.nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module("linear_{}".format(
                i + 1), nn.Linear(self.input_dim, self.input_dim))
            self.layers.add_module("ReLU_{}".format(i + 1), nn.ReLU())
        self.layers.add_module("last_linear", nn.Linear(self.input_dim, 1))

    def forward(self, queries):
        pred_thres = self.layers(queries)
        return pred_thres


