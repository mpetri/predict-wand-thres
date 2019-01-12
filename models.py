import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy import stats
import array
import hyperparams
from util import my_print
import torch.nn.functional as F


class Simple(nn.Module):
    def __init__(self, embed_size, num_layers=hyperparams.default_num_layers):
        super(Simple, self).__init__()
        my_print("Initializing Simple model")
        self.WAND_upper = nn.Embedding(
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
        #print("queries.size()", queries.size())
        # print(queries)
        emb_WAND_Upper = self.WAND_upper(queries[:, :, 0])
        #print("emb_WAND_Upper.size()", emb_WAND_Upper.size())
        emb_Ft = self.Ft(queries[:, :, 1])
        emb_Meanft = self.Meanft(queries[:, :, 2])
        emb_Medft = self.Medft(queries[:, :, 3])
        emb_Minft = self.Minft(queries[:, :, 4])
        emb_Maxft = self.Maxft(queries[:, :, 5])
        emb_Largerft256 = self.Largerft256(queries[:, :, 6])
        emb_Largerft128 = self.Largerft128(queries[:, :, 7])
        emb_Largerft64 = self.Largerft64(queries[:, :, 8])
        emb_Largerft32 = self.Largerft32(queries[:, :, 9])
        emb_Largerft16 = self.Largerft16(queries[:, :, 10])
        emb_Largerft8 = self.Largerft8(queries[:, :, 11])
        emb_Largerft4 = self.Largerft4(queries[:, :, 12])
        emb_Largerft2 = self.Largerft2(queries[:, :, 13])

        #print("emb_WAND_Upper.size()", emb_WAND_Upper.size())
        #print("emb_Meanft.size()", emb_Meanft.size())
        #print("emb_Largerft64.size()", emb_Largerft64.size())

        query_embed = torch.cat((emb_WAND_Upper, emb_Ft, emb_Meanft, emb_Medft, emb_Minft, emb_Maxft, emb_Largerft256, emb_Largerft128,
                                 emb_Largerft64, emb_Largerft32, emb_Largerft16, emb_Largerft8, emb_Largerft4, emb_Largerft2), 2)
        query_embed = query_embed.view(-1, self.input_dim)
        #print("query_embed.size()", query_embed.size())
        pred_thres = self.layers(query_embed)
        #print("pred_thres.size()", pred_thres.size())
        return pred_thres
