#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import neural as n
import numpy as np

#%%]
def deg(x):
    if  x>0:
        return 1
    else:
        return 0
  
Input  = np.loadtxt('input.txt')
Output = np.zeros([len(Input), 2])
with open('output.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'laranja':
            Output[i,0] = 1 
        elif text == 'maca':
            Output[i,1] = 1
    
    
#%%
def f(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)

SR = [[np.array([0,1]), 'Maçã'],
      [np.array([1,0]), 'Laranja']]
neuron_layer = n.layer(2, 3, deg, syntax_resp=SR)
n.training_layer(neuron_layer, Input, Output, eta=lambda a: 10)
#n = training_function(2, deg, Input, Output, eta=f, number_of_seasons=100_000) 