#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:42:23 2019

@author: wang9
"""


import os, sys
import numpy as np
from mir_eval import separation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import soundfile as sf 
import torch
import utils
import glob_constant
from DC_Net_3_128fft_size import DC_net
import librosa
#import sounddevice as sd
from sklearn.cluster import Birch
device = torch.device('cpu')


def norm_fea (feature):
    #return (feature-53.992146)/17.463215
    return (feature-GLOBAL_MEAN)/GLOBAL_STD
    #return (feature-(GLOBAL_MEAN+41.520496)/2)/((GLOBAL_STD+19.247942)/2)
# =============================================================================
# dir_mix = '/home/wang9/narvi/tt/mix/'
# dir_s1 = '/home/wang9/narvi/tt/s1/'
# dir_s2 = '/home/wang9/narvi/tt/s2/'
# =============================================================================
dir_mix = '/home/wang9/narvi/wsj0/min/tt_ff/mix/'
dir_s1 = '/home/wang9/narvi/wsj0/min/tt_ff/s1/'
dir_s2 = '/home/wang9/narvi/wsj0/min/tt_ff/s2/'
# =============================================================================
# dir_mix = '/home/wang9/narvi/pytorch_danish/data/cv/mix/'
# dir_s1 = '/home/wang9/narvi/pytorch_danish/data/cv/s1/'
# dir_s2 = '/home/wang9/narvi/pytorch_danish/data/cv/s2/'
# =============================================================================
# =============================================================================
# dir_mix = '/home/wang9/narvi/LISTENING_TEST/Old/tt_norm/tt_norm_F1F2/mix/'
# dir_s1 = '/home/wang9/narvi/LISTENING_TEST/Old/tt_norm/tt_norm_F1F2/s1/'
# dir_s2 = '/home/wang9/narvi/LISTENING_TEST/Old/tt_norm/tt_norm_F1F2/s2/'
# =============================================================================

no_samples=5

files = os.listdir(dir_mix)
filename=files[:no_samples]
count = 0
mean_std = np.load('/home/wang9/narvi/wsj0/min/tr/global_mean_std_tr_128_fft_size.npz')
GLOBAL_MEAN = mean_std['a']
GLOBAL_STD = mean_std ['b']

model_path = '/home/wang9/narvi/pytorch/net3_128fft_size_1200_units/model.pt'
sdr_arr=np.zeros((no_samples,2))
sir_arr=np.zeros((no_samples,2))
sar_arr=np.zeros((no_samples,2))  
# =============================================================================
# print('-------------------------------------', sys.argv)
# filename=[filename[int(sys.argv[-1])]]
# print('.............................', filename)        
# =============================================================================
for file in filename:
    print('count = ',count)
    mix = utils.get_input(dir_mix+file)
    s1 = utils.get_input(dir_s1+file)
    s2 = utils.get_input(dir_s2+file)
    
    spec0,spec_mix,VAD = utils.get_features(mix)
    
    norm_spec_mix = norm_fea (spec_mix)
    _,spec_s1,_ = utils.get_features(s1)
    _,spec_s2,_ = utils.get_features(s2)
    
    mask_1 = spec_s1 > spec_s2
    mask_2 = spec_s1 <= spec_s2
    
    splitted_spec_mix = utils.split_in_seqs(norm_spec_mix.T,norm_spec_mix.shape[1])
    splitted_VAD=utils.split_in_seqs(VAD.T,spec_mix.shape[1] ).astype('float32')
    #splitted_spec_mix = utils.split_in_seqs(norm_spec_mix.T,200)
    splitted_spec_mix_tensor= torch.from_numpy(splitted_spec_mix)
    
    model = DC_net(glob_constant.input_dim, glob_constant.hidden_dim,splitted_spec_mix_tensor.size(0),
               glob_constant.Embedding,4,glob_constant.dropout)
   
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():    
        hidden = model.init_hidden()
        embedding = model(splitted_spec_mix_tensor, hidden)
        embeddings = embedding.numpy()
        reshape_emb = np.reshape(embeddings,(-1,40))
        
        em_eff=reshape_emb.T*np.reshape(splitted_VAD,(-1))
        em_eff=em_eff.T
        index=np.where(np.any(em_eff,axis=1))
        del_zeros=em_eff[index[0],:]
        x=np.zeros((em_eff.shape[0]))
        y=np.zeros((em_eff.shape[0]))
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(del_zeros)
        estimated_mask=kmeans.labels_
        #estimated_mask=kmeans.labels_
        x[index[0]]=estimated_mask
        y[index[0]]=1-estimated_mask
          
          
        re_estimated_mask_1=np.reshape(x,((-1,utils.NEFF)))
        re_estimated_mask_2=np.reshape(y,((-1,utils.NEFF)))
        
        re_estimated_mask_1 = re_estimated_mask_1.T
        re_estimated_mask_2 = re_estimated_mask_2.T
# =============================================================================
#         re_estimated_mask_1 = re_estimated_mask_1[:,:np.shape(spec0)[1]]
#         re_estimated_mask_2 = re_estimated_mask_2[:,:np.shape(spec0)[1]]
# =============================================================================
        
        s1e=spec0[:,:re_estimated_mask_1.shape[1]]*re_estimated_mask_1
        s2e=spec0[:,:re_estimated_mask_2.shape[1]]*re_estimated_mask_2
        
        # =============================================================================
    # =============================================================================
    #     s1e=mix_spec0*oracle_2
    #     s2e=mix_spec0*oracle_1
    # =============================================================================
        # =============================================================================
        es_s1_tr=librosa.istft(s1e,win_length = utils.win_length, hop_length=utils.hop_length)
        es_s2_tr=librosa.istft(s2e,win_length = utils.win_length, hop_length=utils.hop_length)
        
        #evaluation
        
        s1=s1[:np.min((len(es_s1_tr),len(s1)))]
        s2=s2[:np.min((len(es_s2_tr),len(s2)))]
        
        groundtruth=np.zeros((2,len(s1)))
        estimate=np.zeros((2,len(s1)))
        groundtruth[0,:]=s1
        groundtruth[1,:]=s2
        estimate[0,:]=es_s1_tr
        estimate[1,:]=es_s2_tr
        (sdr, sir, sar, perm)=separation.bss_eval_sources(groundtruth,estimate)
        print('index is ',count,'file name is',file,'sdr', sdr,'sir',sir,'sar',sar)
        sdr_arr[count,:]= sdr
        sir_arr[count,:]= sir
        sar_arr[count,:]= sar
        count+=1
        
        
print('mean of sdr is ',np.mean(sdr_arr ))
print('mean of sir is ',np.mean(sir_arr))
print('mean of sar is ',np.mean(sar_arr )) 
        
# =============================================================================
# 
# output_name='./wsj_tt_fm/out_'+sys.argv[-1]+'.npz'
# 
# np.savez(output_name,SDR=sdr, SIR=sir, SAR=sar)     
# =============================================================================
        
        
