#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import pandas as pd
import numpy as np

np.random.seed(666)
#%% import data
pasta = 'data/'
sonar = pd.read_csv(pasta + 'sonar_csv.csv')

inp     = np.array(np.array(sonar)[:,:-1], dtype=float)
classes = np.array(sonar)[:, -1]
out     = np.array([[0,1] if term=='Mine' else [1,0] for term in classes])

#%% Treinamento
neuron_1 = n.mlp(60,30,2)
erro_1 = v.validate_mlp(neuron_1, inp, out)
#%%
plt.plot(erro_1)
plt.grid()
plt.xlabel('iterações')
plt.ylabel('erro RMS')
plt.title('erro de treinamento Sonar')
plt.xlim([-1,4_000])
plt.savefig('erro_sonar.png')
plt.show()

#%% avaliação
def absolute_aplicate(val):
    _,_,_,resp = neuron_1.aplicate(val)
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

encontrado  = np.array([absolute_aplicate(val) for val in inp])
m_rock      = matriz_de_confusao(encontrado[:,0], out[:, 0])
m_mine      = matriz_de_confusao(encontrado[:,1], out[:, 1])

print(f'---rock---\n{m_rock}') 
print(f'precisão:{m_rock[0,0]/(m_rock[0,0]+m_rock[0,1])}')
print(f'revocação:{m_rock[0,0]/(m_rock[0,0]+m_rock[1,0])}\n')

print(f'---mine---\n{m_mine}') 
print(f'precisão:{m_mine[0,0]/(m_mine[0,0]+m_mine[0,1])}')
print(f'revocação:{m_mine[0,0]/(m_mine[0,0]+m_mine[1,0])}\n')

#%% acurácia
ac = 0
for i,enc in enumerate(encontrado):
    if all(enc == out[i]):
        ac+=1

print(f'Acurácia de {100*ac/len(out) : .2f}%')

#%%  Treino camada
layer_1 = n.layer(2,60)
erro_l  = v.training_layer(layer_1, inp, out)

#%% Teste camada
encontrado  = np.array([layer_1.absolute_aplicate(val)[0] for val in inp])
m_rock      = matriz_de_confusao(encontrado[:,0], out[:, 0])
m_mine      = matriz_de_confusao(encontrado[:,1], out[:, 1])
ac = 0
for i,enc in enumerate(encontrado):
    if all(enc == out[i]):
        ac+=1

print(f'Acurácia de {100*ac/len(out) : .2f}%')
print(f'---rock---\n{m_rock}') 
print(f'precisão:{m_rock[0,0]/(m_rock[0,0]+m_rock[0,1])}')
print(f'revocação:{m_rock[0,0]/(m_rock[0,0]+m_rock[1,0])}\n')

print(f'---mine---\n{m_mine}') 
print(f'precisão:{m_mine[0,0]/(m_mine[0,0]+m_mine[0,1])}')
print(f'revocação:{m_mine[0,0]/(m_mine[0,0]+m_mine[1,0])}\n')