# coding: utf-8
import argparse
import operator as op
from functools import reduce
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim
from scipy import stats
import pandas as pd
from tqdm import tqdm
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from util import my_print, init_log
import hyperparams
import data_loader
import models
from itertools import chain
import gc

parser = argparse.ArgumentParser(description='PyTorch Sentence Similarity')
parser.add_argument('--fasttext_en', type=str, required=True, help='english fasttext model')
parser.add_argument('--fasttext_de', type=str, required=True, help='german fasttext model')
parser.add_argument('--parallel_en', type=str, required=True, help='english parallel sentences')
parser.add_argument('--parallel_de', type=str, required=True, help='german parallel sentences')
parser.add_argument('--mono_en', type=str, required=True, help='english monolingual sentences')
parser.add_argument('--model_file',type=str, required=True, help='training epochs')
parser.add_argument('--output_file',type=str, required=True, help='output file')
parser.add_argument('--device', default="cpu",type=str, required=False, help='compute device')

parser.add_argument('--max_sents', type=int, default=99999999, help='maximum number of sentences')
parser.add_argument('--skip_first_mono', type=int, default=1000000, help='skip the first few mono sentences')
parser.add_argument('--max_mono_sents', type=int, default=1000000, help='maximum number of mono sentences')
parser.add_argument('--max_slen', type=int, default=hyperparams.default_max_slen, help='maximum number of sentences')
parser.add_argument('--batch_size', type=int, default=hyperparams.default_batch_size, help='maximum sentence len')
parser.add_argument('--sent_embed_size', default=hyperparams.default_sembed_size,type=int, required=False, help='sentence embedding size')
parser.add_argument('--margin_loss_margin', default=hyperparams.default_margin,type=int, required=False, help='margin loss margin')
parser.add_argument('--clip', type=float, default=hyperparams.default_gradient_clipping,help='gradient clipping')
parser.add_argument('--neg_samples', type=int, default=hyperparams.default_neg_samples,help='number of negative samples')
parser.add_argument('--lr', type=float, default=hyperparams.default_learning_rate,help='initial learning rate')
parser.add_argument('--epochs', default=hyperparams.default_epochs,type=int, required=False, help='training epochs')


args = parser.parse_args()
init_log(args)
my_print("Parameters:")
for k,v in sorted(vars(args).items()): my_print("\t{0}: {1}".format(k,v))


# Set the random seed manually for reproducibility.
torch.manual_seed(hyperparams.random_seed)

###############################################################################
# Load data
###############################################################################
if torch.cuda.is_available():
    if args.device != "cpu":
        torch.cuda.set_device(int(args.device.split(":")[1]))
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cuda')

else:
    args.device = torch.device('cpu')
my_print("Using torch device:",args.device)

parallel_dataset = data_loader.ParallelSentences(args)

mono_dataset = data_loader.MonoSentences(args,parallel_dataset.embed_en)

with torch.no_grad():

	# Load the saved model.
	with open(args.model_file, 'rb') as f:
		model = torch.load(f,map_location=args.device)

	my_print(model)
	model.eval()

	de_embeds = model.compute_de_embeddings(parallel_dataset,parallel_dataset.test_de)
	de_embeds = de_embeds.to(device=torch.device('cpu'))
	en_embeds = model.compute_en_embeddings(parallel_dataset,parallel_dataset.test_en)
	en_embeds = en_embeds.to(device=torch.device('cpu'))

	num_en_total = len(en_embeds) + len(mono_dataset.sents)
	all_en_embeds = torch.zeros((num_en_total,1,en_embeds.size(2)),device=torch.device('cpu'))
	all_en_embeds[0:len(en_embeds),:,:] = en_embeds

	cur_offset = len(en_embeds)
	chunkSize = 3000
	for i in trange(0, len(mono_dataset.sents), chunkSize):
		mono_en_embeds = model.compute_en_embeddings(parallel_dataset,mono_dataset.sents[i:i+chunkSize])
		all_en_embeds[cur_offset:cur_offset+mono_en_embeds.size(0),:,:] = mono_en_embeds
		cur_offset = cur_offset + chunkSize

	eval_steps = 1000*np.power(2,range(1,16))
	for num_mono in eval_steps:
		cur_embeds = all_en_embeds[0:num_mono]
		total_num_mono = cur_embeds.size(0)
		ranks = []
		for i, de_embed in enumerate(de_embeds):
			comp_ranks,scores = model.compute_rank(de_embed,i,cur_embeds)
			ranks.append(comp_ranks[i])
			my_print("="*89)
			my_print("num_mono {} sent_id {}".format(num_mono,i))
			my_print("\tgold_rank = {} gold_score {}".format(comp_ranks[i],scores[i].item()))
			my_print("\tgold_sent_de = {}".format(parallel_dataset.test_de[i]))
			my_print("\tgold_sent_en = {}".format(parallel_dataset.test_en[i]))
			for j in range(1,11):
				for k, rank in enumerate(comp_ranks):
					if rank == j:
						str_sent = " "
						if k < 1000:
							str_sent = " ".join(parallel_dataset.test_en[k])
						else:
							str_sent = " ".join(mono_dataset.sents[k-1000])
						my_print("\t\trank {} score {} sent '{}'".format(j,scores[k].item(),str_sent))
						break
		mr = np.mean(np.asarray(ranks))
		my_print("num_mono {} total MR = {} ranks {}".format(num_mono,mr,ranks))





