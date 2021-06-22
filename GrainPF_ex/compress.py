#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:38:07 2020

@author: yigongqin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import io as sio
import numpy as np
import time
import h5py

dataname = 'part_data.mat'
dataname = 'data8.mat'
dataset = sio.loadmat(dataname)['data_tiny']

dimo, batch_size = dataset.shape

dataset = dataset/10;

'''
ax1 = plt.subplot(221)
i=0;ax1.plot(dataset[i*nsign:(i+1)*nsign,0])
ax2 = plt.subplot(222)
i=1;ax2.plot(dataset[i*nsign:(i+1)*nsign,0])
ax3 = plt.subplot(223)
i=2;ax3.plot(dataset[i*nsign:(i+1)*nsign,0])
ax4 = plt.subplot(224)
i=3;ax4.plot(dataset[i*nsign:(i+1)*nsign,0])
'''

# NCHW, NCL structure
dat = torch.from_numpy(dataset)

# reset the shape for the input dat
print(dat.shape)
dat = dat.reshape([1,dimo, batch_size])
print(dat.shape)
dat = dat.permute(2,0,1)
print(dat.shape)


# define a autoencoder structure


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(1, 6, 29, stride=12, padding=14), 
            nn.ReLU(),
            nn.Conv2d(6, 2, 15, stride=6, padding=7),
            nn.ReLU(),
        
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(2, 6, 15, stride=6, padding=7, output_padding=4),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, 29, stride=12, padding=14, output_padding=7),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train(model, num_epochs, data_i, data_o ):
    
    learning_rate=1e-3
    
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--

  #  outputs = []
    for epoch in range(num_epochs):

        recon = model(data_i)
        loss = criterion(recon, data_o)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
       # outputs.append((epoch, data, recon),)
        
    return  recon







# set autoencoder
auto = Autoencoder()
auto = auto.double()

# for forward run
#dat_outt= auto.forward(dat)



# for training a autoencoder

max_epochs = 800
dat_outt = train(auto, max_epochs, dat)


embedt= auto.encoder(dat)


print(dat_outt.shape)
dat_outt = dat_outt.permute(1,2,0);
embedt = embedt.permute(1,2,0);
chan,dimn,s = embedt.shape
print(dat_outt.shape)
dat_outt = dat_outt.reshape([dimo, batch_size])
print(dat_outt.shape)

dat_out = dat_outt.detach().numpy()
embed = embedt.detach().numpy()





#sio.savemat(dataname,{'data_tiny':dataset,'dat_out':dat_out,'embed':embed})

conv_params = list(auto.parameters())
print("len(conv_params):", len(conv_params))
print("Filters:", conv_params[0].shape)
print("Biases:", conv_params[1].shape)
print("Filters:", conv_params[2].shape)
print("Biases:", conv_params[3].shape)
print("Filters:", conv_params[4].shape)
print("Biases:", conv_params[5].shape)
print("Filters:", conv_params[6].shape)
print("Biases:", conv_params[7].shape)