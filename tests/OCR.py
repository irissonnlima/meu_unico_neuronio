#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import numpy as np

np.random.seed(666)
#%% import data

pasta  = 'data/'

inp  = np.loadtxt(pasta + 'input_napolitano.txt')
out = np.zeros([len(inp), 3])
with open(pasta + 'output_napolitano.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'morango':
            plt.plot(inp[i,0],inp[i,1], 'kx')
            out[i,0] = 1 
        elif text == 'baunilha':
            plt.plot(inp[i,0],inp[i,1], 'kd')
            out[i,1] = 1
        else:
            plt.plot(inp[i,0],inp[i,1], 'kv')
            out[i,2] = 1
neuron_layer = n.layer(number_of_neurons=3, number_of_elements=2)
v.training_layer(neuron_layer , inp, out, season_to_print=110_000)

N0,N1,N2 = neuron_layer.weights
x0 = np.linspace(3, 13, 10)
x1 = np.linspace(0, 25, 10)
x2 = np.linspace(13, 25, 10)
plt.plot(x0, -(x0*N0[1] + N0[0])/N0[2])
plt.plot(x1, -(x1*N1[1] + N1[0])/N1[2])
plt.plot(x2, -(x2*N2[1] + N2[0])/N2[2])
plt.title('Distribuição das classes no espaço')
plt.xlabel('Parâmetro A')
plt.ylabel('Parâmetro B')
plt.grid()
plt.savefig('classes_napo.png')
plt.show()

#%% Teste de Shuffle eta = 0.1, 0.05, 0.01
neuron_layer05      = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
neuron_layer05n     = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)

neuron_layer01      = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
neuron_layer01n     = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)

neuron_layer005     = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
neuron_layer005n    = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)

neuron_layer001     = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
neuron_layer001n    = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)

error_05            = v.training_layer(neuron_layer05 , inp, out, eta = lambda a: 0.5, shuffle=True ,season_to_print=110_000)
error_05n           = v.training_layer(neuron_layer05n, inp, out, eta = lambda a: 0.5, shuffle=False, season_to_print=110_000)

error_01            = v.training_layer(neuron_layer01 , inp, out, eta = lambda a: 0.1, shuffle=True , season_to_print=110_000)
error_01n           = v.training_layer(neuron_layer01n, inp, out, eta = lambda a: 0.1, shuffle=False, season_to_print=110_000)

error_005           = v.training_layer(neuron_layer005 , inp, out, eta = lambda a: 0.05, shuffle=True , season_to_print=110_000)
error_005n          = v.training_layer(neuron_layer005n, inp, out, eta = lambda a: 0.05, shuffle=False, season_to_print=110_000)

error_001           = v.training_layer(neuron_layer001 , inp, out, eta = lambda a: 0.01, shuffle=True , season_to_print=110_000)
error_001n          = v.training_layer(neuron_layer001n, inp, out, eta = lambda a: 0.01, shuffle=False, season_to_print=110_000)
#%%
print(f'Eta 0.01: {len(error_001)}\t{len(error_001n)}\t razão={len(error_001n)/len(error_001)}')
print(f'Eta 0.05: {len(error_005)}\t{len(error_005n)}\t razão={len(error_005n)/len(error_005)}')
print(f'Eta 0.1 : {len(error_01 )}\t{len(error_01n )}\t razão={len(error_01n) /len(error_01)}')
print(f'Eta 0.5 : {len(error_05 )}\t{len(error_05n )}\t razão={len(error_05n) /len(error_05)}')

#%% Plot Erros Shuffle
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(error_05 [1:], alpha=0.5)
axs[0, 0].plot(error_05n[1:], alpha=0.5)

axs[1, 0].plot(error_01 [1:], alpha=0.5)
axs[1, 0].plot(error_01n[1:], alpha=0.5)

axs[0, 1].plot(error_005 [1:], alpha=0.5)
axs[0, 1].plot(error_005n[1:], alpha=0.5)

axs[1, 1].plot(error_001 [1:], alpha=0.5)
axs[1, 1].plot(error_001n[1:], alpha=0.5)

axs[0, 0].set(ylabel='erro RMS')
axs[1, 0].set(xlabel='iterações', ylabel='erro RMS')
axs[1, 1].set(xlabel='iterações')

axs[0, 0].set_title('eta 0.5')
axs[1, 0].set_title('eta 0.1')
axs[0, 1].set_title('eta 0.05')
axs[1, 1].set_title('eta 0.01')

for i,ax in enumerate(fig.get_axes()):
    ax.label_outer()
    ax.grid()
    if i ==0 or i ==2:
        ax.set_xlim([-500,6_250])
    else:
        ax.set_xlim([-500,11_500])
fig.suptitle('Gráfico com embaralhamento (laranja) e sem (Azul)')
fig.savefig('grid erros por eta.png')
fig.show()

#%% Teste saturação de eta com e sem shuffle
n_no_shuffle         = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
n_shuffle            = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
iteracoes_shuffle    = []
iteracoes_no_shuffle = []

for i in range(1,100):
    # Randomização dos pesos das redes
    n_no_shuffle.weight_randomize()
    n_shuffle   .weight_randomize()
    # Treinamento da rede
    error_ns = v.training_layer(n_no_shuffle, inp, out, eta = lambda a: 1/i, shuffle=False, season_to_print=110_000)
    error_s  = v.training_layer(n_shuffle   , inp, out, eta = lambda a: 1/i, shuffle=True , season_to_print=110_000)
    # append das iterações da rede para o eta
    iteracoes_no_shuffle.append(len(error_ns))
    iteracoes_shuffle   .append(len(error_s) )
    #salvamento dos pesos da rede
    n_no_shuffle.save_weight('w_saturation_no_shuffle')
    n_shuffle   .save_weight('w_saturation_shuffle')
    print(f'----------{i}----------\n')

#%% Plots Do teste de saturação
x = [1/i for i in range(1,101)]
dist_max = max(np.array(iteracoes_no_shuffle)/np.array(iteracoes_shuffle))
indice   = list(np.array(iteracoes_no_shuffle)/np.array(iteracoes_shuffle)).index(dist_max)

plt.plot(x, iteracoes_no_shuffle, label = 'Sem embaralhamento')
plt.plot(x, iteracoes_shuffle, label = 'Com embaralhamento')
plt.plot([x[indice], x[indice]], [iteracoes_shuffle[indice], iteracoes_no_shuffle[indice]], label=f'Maior distância de {dist_max: .2f}x')
plt.grid()
plt.legend()
plt.xlabel('$\eta$ (eta)')
plt.ylabel('n° de iterações')
plt.title('Iterações nescessárias por $\eta$')
plt.savefig('iter_eta_all.png')
plt.plot()

#%% Teste de eta como função de rampa
sns_max     = 100
sns_min     = 1
eta_max     = 1
eta_min     = 0.1
eta_rampa   = lambda sns: ((eta_min-eta_max)/(sns_max-sns_min))*(sns-sns_min) + eta_max

dist_const  = []
for i in range(500):
    n_eta_const = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
    erro_const  = v.training_layer(n_eta_const, inp, out, eta = lambda sns: 0.1, season_to_print=110_000)
    dist_const.append(len(erro_const))
    
N           = 10
avg_rampa   = 0
avg_const   = np.average(dist_const)
for foo in range(N):
    n_eta_rampa = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
    erro_rampa  = v.training_layer(n_eta_rampa, inp, out, eta_rampa, season_to_print=110_000)
    avg_rampa  += len(erro_rampa)
    

print(f'{avg_const} \t {avg_rampa/N}')
#%% Teste eta com exponencial


#%% Plot histograma
plt.hist(dist_const, 15)
plt.grid()
plt.xlabel('quantidade de iterações')
plt.ylabel('N° de individuos')
plt.title('distribuição de iterações para $\eta=0.1$')
plt.savefig('hist_01.png')
plt.show()

#%% Validação OCR
neural = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)
trainx,testx, trainy, testy = train_test_split(inp, out, test_size=0.1, random_state=2)
v.training_layer(neural, trainx, trainy)



