#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:12:14 2018

@author: wang9
"""




import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sb
import os
import glob
# =============================================================================
# data_dir= '/home/wang9/taito/narvi/net5'
# os.chdir(data_dir)
# =============================================================================
num = 1603
file_folder = 'wsj_tt_fm'
sdr= np.zeros((num,2))
sir= np.zeros((num,2))
sar= np.zeros((num,2))
count = 0
for file in glob.glob('./'+file_folder+'/*.npz'):
  #  print(file)
    a=np.load(file)
    sdr[count,:]=a['SDR']
    sir[count,:]=a['SIR']
    sar[count,:]=a['SAR']
    count= count+1
sdr=np.mean(sdr, axis=-1)
sir=np.mean(sir, axis=-1)
sar=np.mean(sar, axis=-1)

print('sdr',np.around(np.mean(sdr),decimals=1),'sir',np.around(np.mean(sir),decimals=1),'sar',np.around(np.mean(sar),decimals=1))

plt.figure(figsize=(20,10))


plt.subplot(1,3, 1), sb.violinplot(sdr, orient='v',  cut=0), plt.xlabel('SDR (dB)'), plt.title(['mean = ',np.around(np.mean(sdr),decimals=1),' dB','median =',np.around(np.median(sdr),decimals=1),' dB'])
plt.subplot(1,3, 2), sb.violinplot(sir, orient='v',  cut=0), plt.xlabel('SIR (dB)'), plt.title(['mean = ',np.around(np.mean(sir),decimals=1),' dB','median =',np.around(np.median(sir),decimals=1),' dB'])
plt.subplot(1,3, 3), sb.violinplot(sar, orient='v',  cut=0), plt.xlabel('SAR (dB)'), plt.title(['mean = ',np.around(np.mean(sar),decimals=1),' dB','median =',np.around(np.median(sar),decimals=1),' dB'])

#sb.violinplot([1,2, 3], [sdr, sir, sar], orient='h',  cut=0)
plt.savefig(str(num)+'_'+file_folder+'.png')



