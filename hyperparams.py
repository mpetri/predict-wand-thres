import numpy as np
import torch

default_learning_rate = 0.001
default_epochs = 20
default_gradient_clipping = 0.25
default_batch_size = 128
default_num_layers = 4
default_embed_size = 32
default_threads = 16
default_max_qry_len = 10
default_dropout = 0.25

random_seed = 12345

num_term_params = 35

# list len buckets [ 2**j for j in range(0,25) ]
const_Ft_buckets = [0] + [2**j for j in range(0, 25)]
const_freq_buckets = [0] + [2**j for j in range(0, 10)]

const_score_buckets = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5,
                       2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 15.0, 25.0, 50.0]

const_doc_len_buckets = [0.01, 0.1, 0.2, 0.5, 0.75,
                         1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0]

num_quantiles = 10
quantiles = torch.Tensor(2 * np.arange(num_quantiles) + 1) / (2.0 * 10)
