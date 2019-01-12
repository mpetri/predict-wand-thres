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
parser.add_argument('--queries', type=str, required=True, help='query data')
parser.add_argument('--dev_queries', type=str,
                    required=True, help='dev query data')
parser.add_argument('--test_queries', type=str,
                    required=False, help='test query data')
parser.add_argument('--batch_size', type=int,
                    default=hyperparams.default_batch_size, help='batch size')
parser.add_argument('--clip', type=float,
                    default=hyperparams.default_gradient_clipping, help='gradient clipping')
parser.add_argument('--lr', type=float,
                    default=hyperparams.default_learning_rate, help='initial learning rate')
parser.add_argument('--embed_size', type=int,
                    default=hyperparams.default_embed_size, help='embedding size')
parser.add_argument('--layers', type=int,
                    default=hyperparams.default_num_layers, help='number of layers')
parser.add_argument('--epochs', default=hyperparams.default_epochs,
                    type=int, required=False, help='training epochs')
parser.add_argument('--output_prefix', default="./",
                    type=str, required=False, help='output prefix')
parser.add_argument('--mse', action='store_true',
                    default=False, help='use MSE error')
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

dataset = data_loader.InvertedIndexData(args, args.queries)
dev_dataset = data_loader.InvertedIndexData(args, args.dev_queries)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# ###############################################################################
# # Build the model
# ###############################################################################
model_file = args.output_prefix + "/" + create_file_name(args) + ".model"
my_print("Writing model to file", model_file)
model = models.Simple(args.embed_size)
model = model.to(device=args.device)
my_print(model)

writer = SummaryWriter(args.output_prefix + "/runs/" + create_file_name(args))


def train(epoch):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    my_print("epoch {} start training {} instances with lr {}".format(
        epoch, len(dataset), lr))

    loss_func = nn.SmoothL1Loss()
    if args.mse == True:
        loss_func = nn.MSELoss()

    with tqdm(total=len(dataloader), unit='batches', desc='train') as pbar:
        losses = []
        for batch_num, batch in enumerate(dataloader):
            optim.zero_grad()
            queries, thres = batch
            scores = model(queries.to(args.device))
            #print(scores, thres)
            loss = loss_func(scores, thres)
            # print(loss)
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


def evaluate(eval_data):
    with torch.no_grad():
        model.eval()
        errors = []
        for qry, thres in eval_data:
            qry = qry.view(1, qry.size(0), qry.size(1))
            pred_thres = model(qry.to(args.device))
            #print("pred {} actual {}".format(pred_thres.item(), thres.item()))
            diff = pred_thres - thres
            errors.append(diff.item())
        return errors


# # Loop over epochs.
lr = args.lr
best_val_loss = None

# # At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        my_print("start epoch {}/{}".format(epoch, args.epochs))
        train(epoch)
        errors = evaluate(dev_dataset)
        val_mean_error = np.mean(np.asarray(errors))
        writer.add_histogram('eval/errors', np.asarray(errors), epoch)
        writer.add_scalar('eval/mean_error', val_mean_error, epoch)
        my_print('-' * 89)
        my_print("epoch {} val mean_error {}".format(epoch, val_mean_error))
        my_print("epoch {} errors {}".format(epoch, errors[:10]))
        my_print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_mean_error < best_val_loss:
            with open(model_file, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_mean_error

except KeyboardInterrupt:
    my_print('-' * 89)
    my_print('Exiting from training early')

writer.close()
