#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:23:37 2019

@author: wang9
"""
import torch.nn as nn
import torch

#from IPython import embed
# Here we define our model as a class
class DC_net(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, embedding_dim,num_layers,dropout):
        super(DC_net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim      
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size= self.hidden_dim, \
                            num_layers= self.num_layers,dropout = self.dropout,batch_first = True)
       # self.drop_out = nn.Dropout(p = self.dropout)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.input_dim * self.embedding_dim)
        self.non_linear = torch.tanh
        self.eps = 1e-8
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros( self.num_layers,self.batch_size,self.hidden_dim).cuda(),
                torch.zeros( self.num_layers,self.batch_size, self.hidden_dim).cuda())
# =============================================================================
#     def init_hidden(self, batch_size):
#         return self.rnn.init_hidden(batch_size)
# =============================================================================
        
    def forward(self, input,hidden):
        # Forward pass through LSTM layer
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
 
       # input has shape of [batch_size, seq_len,input_size] because we set batch_first= True
        self.seq_len = input.size(1)
       # self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(input,hidden) #shape of lstm_out: [batch_size,seq_len,hidden_dim]
       # print('lstm_out shape is ',lstm_out.shape)
      #  y = self.drop_out(lstm_out)
        y = lstm_out
       # y= lstm_out
      #  embed()
        y = y.contiguous().view(self.batch_size * self.seq_len, self.hidden_dim)# reshape lstm_out to [batch_size * seq_len, hidden_dim]
       # print('y shape is ', y.shape)
        emb_V = self.linear(y) # shape is [batch_size * seq_len, input_dim * embedding_dim]
        emb_V = self.non_linear(emb_V)# shape is [batch_size * seq_len, input_dim * embedding_dim]
       # print('embedding matrix shape is ',emb_V.shape)
        emb_V = emb_V.view(self.batch_size,self.seq_len * self.input_dim, self.embedding_dim)# reshape the embedding to [batch_size, seq_len*input_dim, embedding_dim]
      #  print('reshape embdeddng is ', emb_V.shape)
        V_norm = torch.sqrt(torch.sum(torch.pow(emb_V,2), -1))
        V_norm = V_norm.unsqueeze(-1).expand_as(emb_V)
        emb_V_norm = emb_V/(V_norm +self.eps)
      #  print('normalized embdedding is ', emb_V_norm.shape)
       # out = self.non_linear(emb_V_norm) # output_shape is [batch_size, input_dim * seq_len, embedding_dim]
        

        return emb_V_norm
       # return out
# =============================================================================
# input_seq= Variable(torch.randn(glob_constant.batch_size,glob_constant.seq_len,glob_constant.input_dim))
# 
# model = DC_net(glob_constant.input_dim,glob_constant.hidden_dim,glob_constant.batch_size,\
#                glob_constant.Embedding,glob_constant.num_layers,glob_constant.dropout)
# out= model(input_seq)
# =============================================================================
