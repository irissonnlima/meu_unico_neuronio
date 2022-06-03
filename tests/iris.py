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
A = inp[:, 0]
B = inp[:, 1]
C = inp[:, 2]
D = inp[:, 3]

fig, axs = plt.subplots(4, 4)
for i,val in enumerate(out):
    if val[0] == 1:
        tg = 'rx'
    elif val[1] == 1:
        tg = 'bd'
    else:
        tg = 'gv'
    axs[0,0].plot(A[i], A[i],tg, label = None, alpha=0.5)
    axs[0,1].plot(A[i], B[i],tg, label = None, alpha=0.5)
    axs[0,2].plot(A[i], C[i],tg, label = None, alpha=0.5)
    axs[0,3].plot(A[i], D[i],tg, label = None, alpha=0.5)
    
    axs[1,0].plot(B[i], A[i],tg, label = None, alpha=0.5)
    axs[1,1].plot(B[i], B[i],tg, label = None, alpha=0.5)
    axs[1,2].plot(B[i], C[i],tg, label = None, alpha=0.5)
    axs[1,3].plot(B[i], D[i],tg, label = None, alpha=0.5)
    
    axs[2,0].plot(C[i], A[i],tg, label = None, alpha=0.5)
    axs[2,1].plot(C[i], B[i],tg, label = None, alpha=0.5)
    axs[2,2].plot(C[i], C[i],tg, label = None, alpha=0.5)
    axs[2,3].plot(C[i], D[i],tg, label = None, alpha=0.5)
    
    axs[3,0].plot(D[i], A[i],tg, label = None, alpha=0.5)
    axs[3,1].plot(D[i], B[i],tg, label = None, alpha=0.5)
    axs[3,2].plot(D[i], C[i],tg, label = None, alpha=0.5)
    axs[3,3].plot(D[i], D[i],tg, label = None, alpha=0.5)
    
fig.set_size_inches(8, 8)
axs[0, 0].set(ylabel='comprimento da sépala')
axs[1, 0].set(ylabel='largura da sépala')
axs[2, 0].set(ylabel='comprimento da pétala')
axs[3, 0].set(ylabel='largura da pétala')

axs[3, 0].set(xlabel='comprimento da sépala')
axs[3, 1].set(xlabel='largura da sépala')
axs[3, 2].set(xlabel='comprimento da pétala')
axs[3, 3].set(xlabel='largura da pétala')
fig.suptitle('Iris-setosa(vermelho),  Iris-versicolor(azul) e Iris-virginica(verde)')
plt.savefig('iris_16plot.png')
fig.show()
        
#%% Treinamento da rede
neuron  = n.layer(3, 4)
error   = v.training_layer(neuron, inp, out)

#%% Teste direto
e = 0
for i,val in enumerate(inp):
    if not np.all(neuron.absolute_aplicate(val) == out[i]):
        e+=1
print(e/len(inp))

#%% Test bootstrap
avg, std,_ = v.bootstrap_layer(inp, out,
                               reset_function    = lambda : neuron.weight_randomize(),
                               training_function = lambda i, o: v.training_layer(neuron, i, o,number_of_seasons=60_000 ,season_to_print=110_000),
                               aplicate_function = lambda val: neuron.absolute_aplicate(val))

#%% Avaliação individual
inp_setosa = inp[:49]
inp_virginica = inp[99:]
inp_versicolor = inp[49:99]

t_setosa    = [1 if all(neuron.absolute_aplicate(i)[0] == [1,0,0]) else 0 for i in inp_setosa]
t_virginica = [1 if all(neuron.absolute_aplicate(i)[0] == [0,0,1]) else 0 for i in inp_virginica]
t_versicolor= [1 if all(neuron.absolute_aplicate(i)[0] == [0,1,0]) else 0 for i in inp_versicolor]

print(f'Iris Setosa: {100*np.sum(t_setosa)/len(t_setosa) :.2f}%')
print(f'Iris Virginica: {100*np.sum(t_virginica)/len(t_virginica) :.2f}%')
print(f'Iris Versicolor: {100*np.sum(t_versicolor)/len(t_versicolor) :.2f}%')
    
#%% Montagem das matrizes de confusão
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

encontrado  = np.array([neuron.absolute_aplicate(val)[0] for val in inp])
m_setosa    = matriz_de_confusao(encontrado[:,0], out[:, 0])
m_virginica = matriz_de_confusao(encontrado[:,2], out[:, 2])
m_versicolor= matriz_de_confusao(encontrado[:,1], out[:, 1])
   
print(f'---setosa---\n{m_setosa}') 
print(f'precisão:{m_setosa[0,0]/(m_setosa[0,0]+m_setosa[0,1])}')
print(f'revocação:{m_setosa[0,0]/(m_setosa[0,0]+m_setosa[1,0])}\n')

print(f'--virginica--\n{m_virginica}') 
print(f'precisão:{m_virginica[0,0]/(m_virginica[0,0]+m_virginica[0,1])}')
print(f'revocação:{m_virginica[0,0]/(m_virginica[0,0]+m_virginica[1,0])}\n')

print(f'--versicolor--\n{m_versicolor}') 
print(f'precisão:{m_versicolor[0,0]/(m_versicolor[0,0]+m_versicolor[0,1])}')
print(f'revocação:{m_versicolor[0,0]/(m_versicolor[0,0]+m_versicolor[1,0])}\n')


    
    
    
    
