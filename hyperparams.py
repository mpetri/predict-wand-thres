
default_learning_rate = 0.001
default_epochs = 20
default_gradient_clipping = 0.25
default_batch_size = 32
default_num_layers = 4
default_embed_size = 32
default_threads = 16
default_max_qry_len = 10

random_seed = 12345

num_term_params = 13

# list len buckets [ 2**j for j in range(0,25) ]
const_Ft_buckets = [0]+[ 2**j for j in range(0,25) ]
const_freq_buckets = [0]+[ 2**j for j in range(0,10) ]
