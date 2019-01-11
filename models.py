import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy import stats
import array
import hyperparams
from util import my_print
import torch.nn.functional as F


class TwoGRU(nn.Module):
    def __init__(self,input_dim, hidden_dim,dev, num_layers = hyperparams.default_num_layers):
        super(TwoGRU,self).__init__()
        my_print("Initializing TwoGRU model")
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_directions = 2
        self.gru_de = nn.GRU(input_dim, hidden_dim, num_layers,bidirectional=True,batch_first=True)
        self.gru_en = nn.GRU(input_dim, hidden_dim, num_layers,bidirectional=True,batch_first=True)
        self.mapping = nn.Linear(input_dim, input_dim, bias=False)
        self.I = torch.eye(input_dim,device=dev,requires_grad=False)

    def flatten_parameters(self):
        self.gru_de.flatten_parameters()
        self.gru_en.flatten_parameters()

    def forward(self,padded_input,input_lens):
        seq_len = padded_input.size(2)
        batch_size = padded_input.size(0)

        input_de = padded_input[:,0,:,:]
        lens_de = input_lens[:,0,0]
        rotated_input_de = self.mapping(input_de)

        output_de, _ = self.gru_de(rotated_input_de)
        output_de_reshape = output_de.view(-1,seq_len,self.num_directions, self.hidden_dim)
        output_de_backward_top = output_de_reshape[:,0,1,:]
        output_de_forward_top = output_de_reshape[range(batch_size),lens_de,0,:]
        output_de_concat = torch.cat((output_de_forward_top,output_de_backward_top),dim=1).view(batch_size,1,-1)

        input_en = padded_input[:,1:,:,:].contiguous().view(-1,input_de.size(1),input_de.size(2))
        lens_en = input_lens[:,1:,:].contiguous().view(-1)
        output_en, _ = self.gru_en(input_en)
        output_en_reshape = output_en.view(-1,seq_len,self.num_directions, self.hidden_dim)
        output_en_backward_top = output_en_reshape[:,0,1,:].view(batch_size,-1,self.hidden_dim)
        output_en_forward_top = output_en_reshape[range(len(lens_en)),lens_en,1,:].view(batch_size,-1,self.hidden_dim)

        output_en_concat = torch.cat((output_en_forward_top,output_en_backward_top),dim=2)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = torch.neg(cos(output_en_concat,output_de_concat))
        return score

    def compute_de_embeddings(self,dataset,sentences):
        max_len = 0
        for s in sentences:
            max_len = max(len(s),max_len)

        embeds = torch.zeros(len(sentences),max_len,dataset.word_embed_dim,requires_grad=False,device=torch.device('cpu'))
        lens = torch.zeros(len(sentences),requires_grad=False,device=torch.device('cpu'),dtype=torch.int64)

        for sidx,sent in enumerate(sentences):
            lens[sidx] = len(sent) - 1
            embeds[sidx,:,:] = torch.index_select(dataset.word_embeds_de,0,sent)

        embeds = embeds.to(dataset.device)
        lens = lens.to(dataset.device)

        rotated_input_de = self.mapping(embeds)
        output_de, _ = self.gru_de(rotated_input_de)
        output_de_reshape = output_de.view(-1,max_len,self.num_directions, self.hidden_dim)
        output_de_backward_top = output_de_reshape[:,0,1,:]
        output_de_forward_top = output_de_reshape[range(len(sentences)),lens,0,:]
        output_de_concat = torch.cat((output_de_forward_top,output_de_backward_top),dim=1).view(len(sentences),1,-1)

        return output_de_concat

    def compute_en_embeddings(self,dataset,sentences):
        max_len = 0
        for s in sentences:
            max_len = max(len(s),max_len)

        embeds = torch.zeros(len(sentences),max_len,dataset.word_embed_dim,requires_grad=False,device=torch.device('cpu'))
        lens = torch.zeros(len(sentences),requires_grad=False,device=torch.device('cpu'),dtype=torch.int64)

        for sidx,sent in enumerate(sentences):
            lens[sidx] = len(sent) - 1
            embeds[sidx,0:len(sent),:] = torch.index_select(dataset.word_embeds_en,0,sent)

        embeds = embeds.to(dataset.device)
        lens = lens.to(dataset.device)

        output_en, _ = self.gru_en(embeds)
        output_en_reshape = output_en.view(-1,max_len,self.num_directions, self.hidden_dim)
        output_en_backward_top = output_en_reshape[:,0,1,:]
        output_en_forward_top = output_en_reshape[range(len(sentences)),lens,0,:]
        output_en_concat = torch.cat((output_en_forward_top,output_en_backward_top),dim=1).view(len(sentences),1,-1)

        return output_en_concat

    def compute_rank(self,de_embed,correct_idx,en_embeds):
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        de_embed = de_embed.reshape(1,1,de_embed.size(1))
        scores = cos(en_embeds,de_embed)
        ranks = stats.rankdata(scores.cpu().detach().numpy(),method='ordinal')
        return ranks,scores

class SingleGRU(nn.Module):
    def __init__(self,input_dim, hidden_dim,dev, num_layers = hyperparams.default_num_layers):
        super(TwoGRU,self).__init__()
        my_print("Initializing SingleGRU model")
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_directions = 2
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,bidirectional=True,batch_first=True)
        self.mapping = nn.Linear(input_dim, input_dim, bias=False)
        self.I = torch.eye(input_dim,device=dev,requires_grad=False)

    def flatten_parameters(self):
        self.gru.flatten_parameters()

    def forward(self,padded_input,input_lens):
        seq_len = padded_input.size(2)
        batch_size = padded_input.size(0)

        input_de = padded_input[:,0,:,:]
        lens_de = input_lens[:,0,0]
        rotated_input_de = self.mapping(input_de)

        output_de, _ = self.gru(rotated_input_de)
        output_de_reshape = output_de.view(-1,seq_len,self.num_directions, self.hidden_dim)
        output_de_backward_top = output_de_reshape[:,0,1,:]
        output_de_forward_top = output_de_reshape[range(batch_size),lens_de,0,:]
        output_de_concat = torch.cat((output_de_forward_top,output_de_backward_top),dim=1).view(batch_size,1,-1)

        input_en = padded_input[:,1:,:,:].contiguous().view(-1,input_de.size(1),input_de.size(2))
        lens_en = input_lens[:,1:,:].contiguous().view(-1)
        output_en, _ = self.gru(input_en)
        output_en_reshape = output_en.view(-1,seq_len,self.num_directions, self.hidden_dim)
        output_en_backward_top = output_en_reshape[:,0,1,:].view(batch_size,-1,self.hidden_dim)
        output_en_forward_top = output_en_reshape[range(len(lens_en)),lens_en,1,:].view(batch_size,-1,self.hidden_dim)

        output_en_concat = torch.cat((output_en_forward_top,output_en_backward_top),dim=2)
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        score = torch.neg(cos(output_en_concat,output_de_concat))
        return score


    def compute_de_embeddings(self,dataset,sentences):
        max_len = 0
        for s in sentences:
            max_len = max(len(s),max_len)

        embeds = torch.zeros(len(sentences),max_len,dataset.word_embed_dim,requires_grad=False,device=torch.device('cpu'))
        lens = torch.zeros(len(sentences),requires_grad=False,device=torch.device('cpu'),dtype=torch.int64)

        for sidx,sent in enumerate(sentences):
            lens[sidx] = len(sent) - 1
            embeds[sidx,:,:] = torch.index_select(dataset.word_embeds_de,0,sent)

        embeds = embeds.to(dataset.device)
        lens = lens.to(dataset.device)

        rotated_input_de = self.mapping(embeds)
        output_de, _ = self.gru(rotated_input_de)
        output_de_reshape = output_de.view(-1,max_len,self.num_directions, self.hidden_dim)
        output_de_backward_top = output_de_reshape[:,0,1,:]
        output_de_forward_top = output_de_reshape[range(len(sentences)),lens,0,:]
        output_de_concat = torch.cat((output_de_forward_top,output_de_backward_top),dim=1).view(len(sentences),1,-1)

        return output_de_concat

    def compute_en_embeddings(self,dataset,sentences):
        max_len = 0
        for s in sentences:
            max_len = max(len(s),max_len)

        embeds = torch.zeros(len(sentences),max_len,dataset.word_embed_dim,requires_grad=False,device=torch.device('cpu'))
        lens = torch.zeros(len(sentences),requires_grad=False,device=torch.device('cpu'),dtype=torch.int64)

        for sidx,sent in enumerate(sentences):
            lens[sidx] = len(sent) - 1
            embeds[sidx,0:len(sent),:] = torch.index_select(dataset.word_embeds_en,0,sent)

        embeds = embeds.to(dataset.device)
        lens = lens.to(dataset.device)

        output_en, _ = self.gru(embeds)
        output_en_reshape = output_en.view(-1,max_len,self.num_directions, self.hidden_dim)
        output_en_backward_top = output_en_reshape[:,0,1,:]
        output_en_forward_top = output_en_reshape[range(len(sentences)),lens,0,:]
        output_en_concat = torch.cat((output_en_forward_top,output_en_backward_top),dim=1).view(len(sentences),1,-1)

        return output_en_concat

    def compute_rank(self,de_embed,correct_idx,en_embeds):
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        de_embed = de_embed.reshape(1,1,de_embed.size(1))
        scores = cos(en_embeds,de_embed)
        ranks = stats.rankdata(scores.cpu().detach().numpy(),method='ordinal')
        return ranks,scores
