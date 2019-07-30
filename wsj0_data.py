#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:46:39 2019

@author: wang9
"""

import torch
from torch.utils import data
import os
import time
import numpy as np
import h5py
class wDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path):
      'Initialization'
      super(wDataset, self).__init__()
      with h5py.File(data_path, 'r') as self.h5pyLoader:
          
        self.h5pyLoader = h5py.File(data_path, 'r')
        
        self.infeat = self.h5pyLoader['normalized_fea_128_fft_size']#  # input feature, shape: (num_sample, time, freq)
        self.ibm = self.h5pyLoader['label_128_fft_size'] ## ideal binary mask, shape: (num_sample, time*freq, num_spk)
        self.VAD = self.h5pyLoader['VAD_128_fft_size'] # weight threshold matrix, shape: (num_sample, time*freq, 1)     
        self._len = self.infeat.shape[0]

  def __len__(self):
        'Denotes the total number of samples'
        return self._len
    
  def __getitem__(self,index):
      
      infeat_tensor = torch.from_numpy(self.infeat[index])      #'Generates one sample of data'
      ibm_tensor = torch.from_numpy(self.ibm[index]).float()
      VAD_tensor = torch.from_numpy(self.VAD[index]).float() 
    
      return infeat_tensor,  VAD_tensor,ibm_tensor