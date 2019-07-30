#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:54:40 2019

@author: wang9
"""
import soundfile as sf
import librosa
import numpy as np
def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, data.shape[1]))
    return data

def get_features(audio):  
    spec0 = librosa.core.stft(audio,n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    spec = np.abs(spec0)
    spec = np.maximum(spec, np.max(spec) / MIN_AMP)
    spec = 20. * np.log10(spec * AMP_FAC)
# =============================================================================
#     spec = 20. * np.log10(spec )
# =============================================================================
    max_mag = np.max(spec)
    VAD = (spec > (max_mag - THRESHOLD))
    
    return spec0,spec,VAD

def get_input(path_mix):  
    
    mix,_= sf.read(path_mix)
    
    return mix

def get_output(path_s1,path_s2):
    
    
    s1,_ = sf.read(path_s1)
    s2,_ = sf.read(path_s2)
  
    return s1,s2

FRAMES_PER_SAMPLE = 200
AMP_FAC = 10000
MIN_AMP = 10000
THRESHOLD = 40 
n_fft=128
win_length=128
hop_length=64
NEFF = 65  
FRAMES_PER_SAMPLE = 200
EMBBEDDING_D = 40