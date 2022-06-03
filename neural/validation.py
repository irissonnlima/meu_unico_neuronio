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

def training_layer (neuron_layer:n.layer, data:np.array, results:np.array,
                    eta = lambda sns: 0.1, season_to_print:int=1000, 
                    shuffle:bool=True, number_of_seasons:int=100_000, tol:float=1e-3):
    
    rows, cols  = data.shape
    acum_error  = 0.0
    rms_vector  = []

    row_array   = np.arange(0, rows)
    for sns in range(number_of_seasons):
        acum_error *= 0
        for row in row_array:
            phi_v       = neuron_layer.aplicate(np.array([data[row]]))
            error       = results[row] - phi_v
            acum_error += np.sum(0.5*(error)**2)         # RMS ou erro padrão para realizar a correção dos pesos?
            
            Xp  = np.r_[1, data[row]]
            for i in range(neuron_layer.number_of_neurons):
                neuron_layer.weights[i] += error[0,i] * Xp * eta(sns)
        
        acum_error /= rows 
        rms_vector.append(acum_error)
        if shuffle:
            np.random.shuffle(row_array)
        if (sns%season_to_print == 0):
            print(f'\n________________________ Season: {sns} ________________________')
            print(f'RMS in last season: {acum_error:.2e}')
            
        if acum_error<=tol:
            if season_to_print < number_of_seasons:
                print(__converged__)
                print(f'\n________________________ Season: {sns} ________________________')
            print(f'RMS in last season: {acum_error:.2e}')
            break
    return np.array(rms_vector)
 

#%% 
def validate_mlp(mlp, data:np.array, results:np.array,
                    eta = lambda sns: 0.1, season_to_print:int=1000, 
                    shuffle:bool=True, number_of_seasons:int=100_000, tol:float=1e-3):
    
    rms_vect = []
    rows, cols  = data.shape
    row_array   = np.arange(0, rows)
    
    for sns in range(number_of_seasons):
        rms = 0
        for i in row_array:
            Xp = np.array([data[i]])
            Yp = np.array([results[i]]).T
            
            vi, vo, phi_i, phi_o = mlp.aplicate(Xp)
            error = Yp - phi_o
            rms  += np.sum(error**2)
            
            # Output layer
            delta_o = error*mlp.d_activation_function(vo)
            
            # Hidden layer
            w_o_kj  = mlp.output[:, 1:]
            delta_h = mlp.d_activation_function(vi).T * delta_o.T.dot(w_o_kj)
            
            # Treinamento
            mlp.output += eta(sns) * delta_o.dot(np.r_[[[1]], phi_i].T)
            mlp.hidden += eta(sns) * delta_h.T.dot(np.r_[[[1]], Xp.T].T)
        
        rms /= len(data)  
        rms_vect.append(rms)
        if shuffle:
            np.random.shuffle(row_array)
        if (sns%season_to_print == 0):
            print(f'\n________________________ Season: {sns} ________________________')
            print(f'RMS in last season: {rms:.2e}')
        
        if rms<=tol:
            if season_to_print < number_of_seasons:
                print(__converged__)
                print(f'\n________________________ Season: {sns} ________________________')
            print(f'RMS in last season: {rms:.2e}')
            break
    return rms_vect
        

#%%           
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
        results_test    = np.array([results[i] for i in test_index])

        training_function(data_train, results_train)
        erro_iteration  = 0
        for i,res in enumerate(data_test):
            if not np.all(aplicate_function(res) == results_test[i]):
                erro_iteration+=1
        erro.append(erro_iteration/len(results_test))
        print(f'~~> erro bootstrap: {erro_iteration/len(results_test)}')
    return np.average(erro), np.std(erro), np.array(erro)
        
        
        











    