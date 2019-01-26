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
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='PyTorch WAND Thres predictor')
parser.add_argument('--data_dir', type=str,
                    required=True, help='query data dir')
parser.add_argument('--batch_size', type=int,
                    default=hyperparams.default_batch_size, help='batch size')
parser.add_argument('--clip', type=float,
                    default=hyperparams.default_gradient_clipping, help='gradient clipping')
parser.add_argument('--lr', type=float,
                    default=hyperparams.default_learning_rate, help='initial learning rate')
parser.add_argument('--layers', type=int,
                    default=hyperparams.default_num_layers, help='number of layers')
parser.add_argument('--k', type=int, required=True, help='prediction depth')
parser.add_argument('--epochs', default=hyperparams.default_epochs,
                    type=int, required=False, help='training epochs')
parser.add_argument('--quantile', type=float,
                    default=0.5, help='quantile')
parser.add_argument('--device', default="cpu", type=str,
                    required=False, help='compute device')
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true')
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

train_file = args.data_dir + "/train.json"
if args.debug == True:
    train_file = args.data_dir + "/debug.json"
dev_file = args.data_dir + "/dev.json"

dataset = data_loader.InvertedIndexData(args, train_file)
dev_dataset = data_loader.InvertedIndexData(args, dev_file)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

output_prefix = args.data_dir + "/models/"

# ###############################################################################
# # Build the model
# ###############################################################################
model_file = output_prefix + "/" + create_file_name(args) + ".model"
my_print("Writing model to file", model_file)
model = models.MLP(args.layers)
model = model.to(device=args.device)
my_print(model)

writer = SummaryWriter(output_prefix + "/runs/" + create_file_name(args))

huber_loss = nn.SmoothL1Loss(reduction='none')

quantiles = torch.tensor([[args.quantile]]).view(1, -1).float().to(args.device)


def quantile_loss(x, y):
    diff = x - y
    loss = huber_loss(x, y) * (quantiles -
                               (diff.detach() < 0).float()).abs()
    loss = loss.mean().abs()
    return loss


def quantile_loss_eval(x, y):
    x = torch.from_numpy(x).float().to(args.device)
    y = torch.from_numpy(y).float().to(args.device)
    diff = x - y
    loss = huber_loss(x, y) * (quantiles -
                               (diff.detach() < 0).float()).abs()
    loss = loss.mean().abs()
    return loss


def train(epoch):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    my_print("epoch {} start training {} instances with lr {}".format(
        epoch, len(dataset), lr))

    with tqdm(total=len(dataloader), unit='batches', desc='train') as pbar:
        losses = []
        for batch_num, batch in enumerate(dataloader):
            optim.zero_grad()
            queries, thres = batch
            scores = model(queries.to(args.device))
            loss = quantile_loss(scores, thres.to(args.device))
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
        pred = []
        actual = []
        for qry, thres in eval_data:
            qry = qry.view(1, qry.size(0))
            pred_thres = model(qry.to(args.device))
            diff = pred_thres - thres.to(args.device)
            pred.append(pred_thres.item())
            actual.append(thres.item())
        return np.asarray(pred), np.asarray(actual)


# # Loop over epochs.
lr = args.lr
best_val_loss = None

# # At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        my_print("start epoch {}/{}".format(epoch, args.epochs))
        train(epoch)
        pred, actual = evaluate(dev_dataset)
        q_eval_loss = quantile_loss_eval(pred, actual)
        errors = pred - actual
        writer.add_histogram('eval/errors', np.asarray(errors), epoch)
        writer.add_scalar('eval/q_eval_loss', q_eval_loss, epoch)
        my_print('-' * 89)
        my_print("epoch {} val q_eval_loss {}".format(epoch, q_eval_loss))
        my_print("epoch {} errors {}".format(epoch, errors[:10]))
        my_print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or q_eval_loss < best_val_loss:
            with open(model_file, 'wb') as f:
                torch.save(model, f)
            best_val_loss = q_eval_loss

except KeyboardInterrupt:
    my_print('-' * 89)
    my_print('Exiting from training early')

writer.close()
