import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy import stats
import array
import hyperparams
from util import my_print
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers=hyperparams.default_num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.input_dim = hyperparams.default_max_qry_len * hyperparams.num_term_params
        self.layers = torch.nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module("Dropout_{}".format(
                i + 1), nn.Dropout(p=hyperparams.default_dropout))
            self.layers.add_module("linear_{}".format(
                i + 1), nn.Linear(self.input_dim, self.input_dim))
            self.layers.add_module("ReLU_{}".format(i + 1), nn.ReLU())
            self.layers.add_module("BatchNorm1d_{}".format(
                i + 1), nn.BatchNorm1d(self.input_dim))
        self.layers.add_module("last_linear", nn.Linear(self.input_dim, 1))

    def forward(self, queries):
        pred_thres = self.layers(queries)
        return pred_thres
