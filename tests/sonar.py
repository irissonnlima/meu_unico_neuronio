#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:44:02 2022

@author: widmark
"""

import neural.neural as n
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%% import data

pasta = '/home/widmark/Documentos/Code/machine_learning/meu_unico_neuronio/data/'


sonar = pd.read_csv(pasta + 'sonar_csv.csv')

atributos_sonar = np.array(sonar)[:,:-1]
outputs = np.array(sonar)[:,-1]

#%% função de ativação e taxa de aprendizado

def tanh(v):
    return np.tanh(v/2)

def sigmoid(v):
    return 1/(1+np.exp(-v))

#%% trainamento
    