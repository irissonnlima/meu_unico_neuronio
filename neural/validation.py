#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.utils import resample
import neural.neural as n
import numpy as np


__converged__  = '░█████╗░░█████╗░███╗░░██╗██╗░░░██╗███████╗██████╗░░██████╗░███████╗██████╗░██╗\n'
__converged__ += '██╔══██╗██╔══██╗████╗░██║██║░░░██║██╔════╝██╔══██╗██╔════╝░██╔════╝██╔══██╗██║\n'
__converged__ += '██║░░╚═╝██║░░██║██╔██╗██║╚██╗░██╔╝█████╗░░██████╔╝██║░░██╗░█████╗░░██║░░██║██║\n'
__converged__ += '██║░░██╗██║░░██║██║╚████║░╚████╔╝░██╔══╝░░██╔══██╗██║░░╚██╗██╔══╝░░██║░░██║╚═╝\n'
__converged__ += '╚█████╔╝╚█████╔╝██║░╚███║░░╚██╔╝░░███████╗██║░░██║╚██████╔╝███████╗██████╔╝██╗\n'
__converged__ += '░╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝░╚═════╝░╚══════╝╚═════╝░╚═╝\n'

#%% validation functions
def training_layer (neuron_layer:n.layer, data:np.array, results:np.array,
                    eta = lambda ssns: 0.1, season_to_print:int=1000, 
                    shuffle:bool=True, number_of_seasons:int=100_000, tol:float=1e-3):
    
    rows, cols  = data.shape
    acum_error  = 0.0
    rms_vector  = []

    row_array   = np.arange(0, rows)
    for ssns in range(number_of_seasons):
        acum_error *= 0
        for row in row_array:
            phi_v       = neuron_layer.aplicate(data[row])
            error       = results[row] - phi_v
            acum_error += sum(0.5*(error)**2)         # RMS ou erro padrão para realizar a correção dos pesos?

            for i in range(neuron_layer.number_of_neurons):
                Xp  = np.r_[1, data[row]]
                wn1 = neuron_layer.neurons[i].weight + error[i] * Xp * eta(ssns)
                neuron_layer.neurons[i].weight_atribute( wn1 )
        
        acum_error /= rows 
        rms_vector.append(acum_error)
        if shuffle:
            np.random.shuffle(row_array)
        if (ssns%season_to_print == 0):
            print(f'\n________________________ Season: {ssns} ________________________')
            print(f'RMS in last season: {acum_error:.2e}')
            
        if acum_error<=tol:
            print(__converged__)
            print(f'\n________________________ Season: {ssns} ________________________')
            print(f'RMS in last season: {acum_error:.2e}')
            break
    return np.array(rms_vector)


def training_mlp (multilayer_perceptron:n.mlp, data:np.array, results:np.array,
                  eta:any=lambda ssns: 0.1, season_to_print:int=1000, 
                  shuffle:bool=True, number_of_seasons:int=100_000, tol:float=1e-3):
    
    rows, cols  = data.shape

    row_array   = np.arange(0, rows)
    for ssns in range(number_of_seasons):
        for row in row_array:
            phi, dphi   = multilayer_perceptron.multi_answer(data[row])
            #=== Caso 1 ===
            for neuron in multilayer_perceptron.output_layer.neurons:
                erro_j      = results[row] - phi[-1]
                delta_j     = erro_j * dphi[-1]
                Delta_w_ji  = eta(ssns) * delta_j * data[row]
                w_ji_np1    = multilayer_perceptron.output_layer.
            
            
def bootstrap_layer (data:np.array, results:np.array, reset_function:any,
                     training_function:any, aplicate_function:any, N:int=10):
    erro = []
    for foo in range(N):
        print(f'======================= BOOTSTRAP {foo} =======================')
        reset_function()
        train_index = resample(list(range(len(data))))
        test_index  = list(filter(lambda x: x!=-1,
                      [i if not i in train_index else -1 for i in range(len(data))]))
        
        data_train      = np.array([data[i] for i in train_index])
        results_train   = np.array([results[i] for i in train_index])
        
        data_test       = np.array([data[i] for i in test_index])
        results_test    = np.array([data[i] for i in test_index])

        training_function(data_train, results_train)
        erro_iteration  = 0
        for i,res in enumerate(data_test):
            if not all(aplicate_function(res) == results_test[i]):
                erro_iteration+=1
        erro.append(erro_iteration/len(results_test))
        print(f'~~> erro bootstrap: {erro_iteration/len(results_test)}')
    return np.average(erro), np.std(erro), np.array(erro)
        
        
        
        












    