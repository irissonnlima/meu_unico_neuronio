#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import neural.neural as n
import numpy as np


__converged__  = '░█████╗░░█████╗░███╗░░██╗██╗░░░██╗███████╗██████╗░░██████╗░███████╗██████╗░██╗\n'
__converged__ += '██╔══██╗██╔══██╗████╗░██║██║░░░██║██╔════╝██╔══██╗██╔════╝░██╔════╝██╔══██╗██║\n'
__converged__ += '██║░░╚═╝██║░░██║██╔██╗██║╚██╗░██╔╝█████╗░░██████╔╝██║░░██╗░█████╗░░██║░░██║██║\n'
__converged__ += '██║░░██╗██║░░██║██║╚████║░╚████╔╝░██╔══╝░░██╔══██╗██║░░╚██╗██╔══╝░░██║░░██║╚═╝\n'
__converged__ += '╚█████╔╝╚█████╔╝██║░╚███║░░╚██╔╝░░███████╗██║░░██║╚██████╔╝███████╗██████╔╝██╗\n'
__converged__ += '░╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝░╚═════╝░╚══════╝╚═════╝░╚═╝\n'

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