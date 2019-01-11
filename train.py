# coding: utf-8
import argparse
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
from util import my_print, init_log, create_file_name
import hyperparams
import data_loader
import models
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--terms', type=str, required=True, help='term data')
parser.add_argument('--queries', type=str, required=True, help='query data')
parser.add_argument('--thresholds', type=str,
                    required=True, help='threshold data')
parser.add_argument('--batch_size', type=int,
                    default=hyperparams.default_batch_size, help='batch size')
parser.add_argument('--clip', type=float,
                    default=hyperparams.default_gradient_clipping, help='gradient clipping')
parser.add_argument('--lr', type=float,
                    default=hyperparams.default_learning_rate, help='initial learning rate')
parser.add_argument('--epochs', default=hyperparams.default_epochs,
                    type=int, required=False, help='training epochs')
parser.add_argument('--output_prefix', default="./",
                    type=str, required=False, help='output prefix')
parser.add_argument('--device', default="cpu", type=str,
                    required=False, help='compute device')
args = parser.parse_args()
init_log(args)
torch.set_num_threads(hyperparams.default_threads)
my_print("Parameters:")
for k, v in sorted(vars(args).items()):
    my_print("\t{0}: {1}".format(k, v))

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
my_print("Using torch device:", args.device)

dataset = data_loader.InvertedIndexData(args)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# ###############################################################################
# # Build the model
# ###############################################################################
model_file = args.output_prefix + "/" + create_file_name(args) + ".model"
my_print("Writing model to file", model_file)
model = models.Simple()
model = model.to(device=args.device)
my_print(model)

writer = SummaryWriter(args.output_prefix + "/runs/" + create_file_name(args))


def train(epoch):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    my_print("epoch {} start training {} instances with lr {}".format(
        epoch, len(dataset), lr))

    loss_func = nn.SmoothL1Loss()

    with tqdm(total=len(dataloader), unit='batches', desc='train') as pbar:
        losses = []
        for batch_num, batch in enumerate(dataloader):
            optim.zero_grad()
            queries, thres = batch
            scores = model(queries.to(args.device))
            loss = loss_func(scores, thres)
            writer.add_scalar('loss/total', loss.item(), batch_num)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            pbar.set_postfix(loss=np.mean(losses[-128:]))
            writer.add_scalar('loss/mean-over-128',
                              np.mean(losses[-128:]), batch_num)
            pbar.update(1)
        for name, param in model.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, batch_num)
        pbar.close()


def evaluate(dataset):
    with torch.no_grad():
        model.eval()
        ranks = []
        de_embeds = model.compute_de_embeddings(dataset, dataset.dev_de)
        en_embeds = model.compute_en_embeddings(dataset, dataset.dev_en)
        for i, de_embed in enumerate(de_embeds):
            cmp_ranks, _ = model.compute_rank(de_embed, i, en_embeds)
            ranks.append(cmp_ranks[i])
        return ranks


# # Loop over epochs.
lr = args.lr
best_val_loss = None

# # At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        my_print("start epoch {}/{}".format(epoch, args.epochs))
        train(epoch)
        ranks = evaluate(parallel_dataset)
        writer.add_histogram('eval/ranks', np.asarray(ranks), epoch)
        val_mr = np.mean(np.asarray(ranks))
        writer.add_scalar('eval/mean_rank', val_mr, epoch)
        my_print('-' * 89)
        my_print("epoch {} val MR {}".format(epoch, val_mr))
        my_print("epoch {} ranks {}".format(epoch, ranks))
        my_print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_mr < best_val_loss:
            with open(model_file, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_mr

except KeyboardInterrupt:
    my_print('-' * 89)
    my_print('Exiting from training early')

writer.close()
