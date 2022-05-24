#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:07:13 2022

@author: machina
"""
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import numpy as np

#%% import data

pasta  = 'data/'

inp  = np.loadtxt(pasta + 'input_napolitano.txt')
out = np.zeros([len(inp), 3])
with open(pasta + 'output_napolitano.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'morango':
            plt.plot(inp[i,0],inp[i,1], 'ro')
            out[i,0] = 1 
        elif text == 'baunilha':
            plt.plot(inp[i,0],inp[i,1], 'yo')
            out[i,1] = 1
        else:
            plt.plot(inp[i,0],inp[i,1], 'ko')
            out[i,2] = 1
plt.title('Classes')
plt.xlabel('Parâmetro A')
plt.ylabel('Parâmetro B')
plt.grid()
plt.show()
#%% criação da rede
neuron_layer01   = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.sigmoid)
neuron_layer005   = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.sigmoid)
neuron_layer001   = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.sigmoid)
error_01            = v.training_layer(neuron_layer01, inp, out, eta = lambda a: 0.1)
error_01n           = v.training_layer(neuron_layer01, inp, out, eta = lambda a: 0.1, shuffle=False)

error_005           = v.training_layer(neuron_layer005, inp, out, eta = lambda a: 0.05)
error_005n          = v.training_layer(neuron_layer005, inp, out, eta = lambda a: 0.05, shuffle=False)

error_001           = v.training_layer(neuron_layer001, inp, out, eta = lambda a: 0.01)
error_001n          = v.training_layer(neuron_layer001, inp, out, eta = lambda a: 0.01, shuffle=False)

#%% Plot Erros

plt.plot(error_01 [1:] , label = 'Erro para eta=0.1' , alpha=0.5)
plt.plot(error_01n[1:] , label = 'Erro para eta=0.1 sem shuffle' , alpha=0.5)

plt.plot(error_001 [1:], label = 'Erro para eta=0.01', alpha=0.5)
plt.plot(error_001n[1:], label = 'Erro para eta=0.01 sem shuffle', alpha=0.5)

plt.plot(error_005 [1:], label = 'Erro para eta=0.05', alpha=0.5)
plt.plot(error_005n[1:], label = 'Erro para eta=0.05 sem shuffle', alpha=0.5)
plt.xlabel('número de iterações')
plt.ylabel('Erro RMS')
plt.legend()
plt.show()

#%% Testar saturação de eta
neuron_layer   = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.sigmoid)
iter = []
for i in range(1,200):
    neuron_layer.weight_randomize()
    error = v.training_layer(neuron_layer, inp, out, eta = lambda a: 1/i, seas)
    iter.append(len(error))