#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import numpy as np
#%% import dados
  
pasta  = 'data/'

Input  = np.loadtxt(pasta + 'input.txt')
Output = np.zeros([len(Input), 2])
with open(pasta + 'output.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'laranja':
            Output[i,0] = 1 
        elif text == 'maca':
            Output[i,1] = 1

#%% treinamento
SR = [[np.array([0,1]), 'Maçã'],
      [np.array([1,0]), 'Laranja']]

neuron_layer = n.layer(2, 3, n.deg, syntax_resp=SR)
error = v.training_layer(neuron_layer, Input, Output, eta = lambda a: 0.001)

#%% Plotagem
plt.plot(error)
  