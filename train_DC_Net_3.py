#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:31:47 2019

@author: wang9
"""
import glob_constant
from DC_Net_3_128fft_size import DC_net
from torch.autograd import Variable
import torch
import numpy as np
from torch.utils.data import DataLoader
from wsj0_data import wDataset
import torch.optim as optim
from IPython import embed
import time
import argparse
import matplotlib.pyplot as plt
#net = './net3/'
# define data loaders
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

training_data_path = '/home/wang9/narvi/wsj0/min/tr/data_128_fft_size.h5'
cv_data_path = '/home/wang9/narvi/wsj0/min/cv/data_128_fft_size.h5'


params = {'batch_size': glob_constant.batch_size,
          'shuffle': True,
          'num_workers': 1,
          'drop_last':True}
tr_wDataset =wDataset(training_data_path)
training_generator = DataLoader(tr_wDataset,**params) 
                          
cv_wDataset =wDataset(cv_data_path)
validation_loader = DataLoader(cv_wDataset,**params) 

parser = argparse.ArgumentParser(description='DCNet')
parser.add_argument('--log-step', type=int, default=100,
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--val-save', type=str,  default='model.pt',
                   help='path to save the best model')
args, _ = parser.parse_known_args()
def loss_fn(Y,V,VAD):
    """
    V: (B, T*F, D)
    Y : ()(B, T*F, nspk)
    VAD (B, T, F) -> VAD (B, T*F, 1)

    """
   # VAD = VAD.view(model.batch_size,model.seq_len*model.input_dim,1)
    #VAD = VAD.float()
   # embed()
    #Y = Y.float()
    
    V = V * VAD.expand(VAD.size(0), VAD.size(1), V.size(-1))       
    Y = Y * VAD.expand(VAD.size(0), VAD.size(1), Y.size(-1))
    
    loss = torch.pow(torch.norm(torch.bmm(torch.transpose(V, 1, 2), V)), 2) - \
                            2 * torch.pow(torch.norm(torch.bmm(torch.transpose(V, 1, 2),  Y)), 2) + \
                                torch.pow(torch.norm(torch.bmm(torch.transpose(Y, 1, 2), Y)), 2)
      
    loss = loss/(model.batch_size*model.seq_len*model.input_dim*glob_constant.nskp)     
  #  print('loss is ', loss)               
    return loss


# =============================================================================
# model = DC_net(glob_constant.input_dim,glob_constant.hidden_dim,glob_constant.batch_size,
#                glob_constant.Embedding,glob_constant.num_layers,
#                glob_constant.dropout)
# =============================================================================

model = DC_net(glob_constant.input_dim, glob_constant.hidden_dim, glob_constant.batch_size, 
               glob_constant.Embedding,glob_constant.num_layers,glob_constant.dropout)
print(model)
optimizer =optim.Adam(model.parameters(), lr=0.001)
scheduler  = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
scheduler.step()
print('-----------start training')

def train(epoch):
    start_time = time.time()
    model.train()
    train_loss = 0.
    #embed()
    for batch_idx, data in enumerate(training_generator):
       # embed()
      #  print('I am here')
        
# =============================================================================
#         d_0=data[0].cuda()
#         d_1=data[1].float().cuda()
#         d_2=data[2].cuda()
#         d_0.requires_grad=True
#         d_1.requires_grad=True
#      #   d_2.requires_grad=True
#         
# =============================================================================
      
        batch_infeat = Variable(data[0]).contiguous()
        
        batch_VAD = Variable(data[1]).contiguous()

        batch_Y = Variable(data[2]).contiguous()
      #  print(time.time() - start_time)
     #   print(time.time() - start_time)

        batch_infeat = batch_infeat.to(device)
        batch_VAD = batch_VAD.to(device) 
        batch_Y = batch_Y.to(device)
        
        # training
        hidden = model.init_hidden()
        optimizer.zero_grad()
        
        embedding = model(batch_infeat,hidden)
       # embed()
        #print(time.time() - start_time)
        loss = loss_fn(batch_Y,embedding,batch_VAD)
        #print(time.time() - start_time)
        #set_trace()
        loss.backward()
        #print(time.time() - start_time)
        #if epoch ==2:
        #    embed()
        train_loss += loss.data.item()
        optimizer.step()
        #print(time.time() - start_time)
      #  embed()
            # output logs
                #print('the ',batch_idx,'iteration loss is ', train_loss)
        if (batch_idx+1) % 100 == 0:
            elapsed = time.time() - start_time
            
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(training_generator),
                elapsed * 1000 / (batch_idx+1), loss ))
   
    
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))
       
    return train_loss
def validate(epoch):
    start_time = time.time()
    model.eval()
    validation_loss = 0.
    
    # data loading

    for batch_idx, data in enumerate(validation_loader):
        #if data[0].shape[0] == glob_constant.batch_size:
            
        batch_infeat = Variable(data[0]).contiguous()    
        batch_VAD = Variable(data[1]).contiguous()
        batch_Y = Variable(data[2]).contiguous()
    
        batch_infeat = batch_infeat.to(device)
        batch_VAD = batch_VAD.to(device) 
        batch_Y = batch_Y.to(device)
        
        with torch.no_grad():
             hidden = model.init_hidden()
             embedding = model(batch_infeat, hidden)
             loss = loss_fn(batch_Y,embedding,batch_VAD)  
             validation_loss += loss.data.item()
    #print('the ',batch_idx,'iteration val_loss is ', validation_loss)
    validation_loss /= (batch_idx+1)
   # embed()
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss     
    
   
training_loss = []
validation_loss = []
decay_cnt = 0
for epoch in range(1, 101):
    model.cuda()
    print('this is epoch', epoch)
    training_loss.append(train(epoch))
                                       # Call training
    validation_loss.append(validate(epoch)) # call validation
    print('-' * 99)
    print('after epoch', epoch, 'training loss is ', training_loss, 'validation loss is ', validation_loss)
    if training_loss[-1] == np.min(training_loss):
        print('      Best training model found.')
        print('-' * 99)
    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(args.val_save, 'wb') as f:
            torch.save(model.cpu().state_dict(), f)
            
            print('      Best validation model found and saved.')
            print('-' * 99)

        #torch.save(model, os.path.join(os.path.dirname(args.val_save), 'alt_' + os.path.basename(args.val_save))) # save in alternative format 
        
    decay_cnt += 1
    # lr decay
    #if np.min(validation_loss) not in validation_loss[-3:] and decay_cnt >= 3:
    if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
        scheduler.step()
        decay_cnt = 0
        print('      Learning rate decreased.')
        print('-' * 99)
plt.plot(training_loss,'r')
#plt.hold(True)
plt.plot(validation_loss,'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig('./loss-vs-val_loss_200')
