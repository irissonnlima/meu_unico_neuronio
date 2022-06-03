#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import pandas as pd
import numpy as np

np.random.seed(666)
#%% Import dados
data = np.array(pd.read_csv('data/iris.data'))
inp = np.array([[float(val[0]), float(val[1]), float(val[2]), float(val[3])] for val in data[:,:-1]])
out = []
for val in data[:, -1]:
    if   val == 'Iris-setosa':
        out.append([1, 0, 0])
    elif val == 'Iris-versicolor':
        out.append([0, 1, 0])
    elif val == 'Iris-virginica':
        out.append([0, 0, 1])
    else:
        print('ERRO DE IMPORT!')
        break
out = np.array(out)

#%%
neuron_1 = n.mlp(4,4,3)
neuron_2 = n.mlp(4,6,3)

erro_1 = v.validate_mlp(neuron_1, inp, out)
erro_2 = v.validate_mlp(neuron_2, inp, out)

#%%
N = 40_000
plt.plot(np.arange(1,len(erro_1)),erro_1)
plt.show()
#%%
def absolute_aplicate(n, val):
    _,_,_,resp = n.aplicate(val)
    r = []
    for i in resp:
        if i > 0.5:
            r.append(1)
        else:
            r.append(0)
    return r

def matriz_de_confusao(encontrado, esperado)->np.array:
    matrix = np.zeros([2,2])
    for i,_ in enumerate(encontrado):
        if   encontrado[i] == 1 and esperado[i] == 1:
            matrix[0, 0] += 1
        elif encontrado[i] == 1 and esperado[i] == 0:
            matrix[0, 1] += 1
        elif encontrado[i] == 0 and esperado[i] == 1:
            matrix[1, 0] += 1
        elif encontrado[i] == 0 and esperado[i] == 0:
            matrix[1, 1] += 1           
    return matrix

for n in [neuron_1, neuron_2]:
    encontrado  = np.array([absolute_aplicate(n, val) for val in inp])
    m_setosa    = matriz_de_confusao(encontrado[:,0], out[:, 0])
    m_virginica = matriz_de_confusao(encontrado[:,2], out[:, 2])
    m_versicolor= matriz_de_confusao(encontrado[:,1], out[:, 1])
    ac = 0
    for i,enc in enumerate(encontrado):
        if all(enc == out[i]):
            ac+=1

    print(f'Acurácia de {100*ac/len(out) : .2f}%')
    print(f'---setosa---\n{m_setosa}') 
    print(f'precisão:{m_setosa[0,0]/(m_setosa[0,0]+m_setosa[0,1])}')
    print(f'revocação:{m_setosa[0,0]/(m_setosa[0,0]+m_setosa[1,0])}\n')
    
    print(f'--virginica--\n{m_virginica}') 
    print(f'precisão:{m_virginica[0,0]/(m_virginica[0,0]+m_virginica[0,1])}')
    print(f'revocação:{m_virginica[0,0]/(m_virginica[0,0]+m_virginica[1,0])}\n')
    
    print(f'--versicolor--\n{m_versicolor}') 
    print(f'precisão:{m_versicolor[0,0]/(m_versicolor[0,0]+m_versicolor[0,1])}')
    print(f'revocação:{m_versicolor[0,0]/(m_versicolor[0,0]+m_versicolor[1,0])}\n')