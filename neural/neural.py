#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def __signal__(x):
    if x >= 0:
        return '+'
    else:
        return '-'
    
    
class neuron:
    def __init__(self, number_of_elements:int, activation_function:any):
        self.number_of_elements = number_of_elements
        self.weight             = np.zeros(number_of_elements)
        self.phi                = activation_function

    def __repr__(self):
        representation = '+' + '-'*14 + '+\n'
        for i in range(self.number_of_elements):
            representation += f'|w{i} = {__signal__(self.weight[i])}{abs(self.weight[i]):.2e}|\n'
        representation += '+' + '-'*14 + '+\n'
        return representation
    
    def weight_atribute(self, weight_array:np.array):
        for i, w in enumerate(weight_array):
            self.weight[i] = w
            
    def randomize_weight(self, maximum:int = 1, minimum:int=-1):
        assert maximum>minimum, "Máximo SEMPRE deve ser maior que o mínimo"
        
        values = (maximum-minimum)*maximum*np.random.random(self.number_of_elements) + minimum
        for i,w in enumerate(values):
            self.weight[i] = w
    
    def aplicate(self, data):
        Xp = np.r_[1, data]
        v  = np.dot(Xp, self.weight.T)
        return self.phi(v)
        

class layer:
    def __init__(self, number_of_neurons:int, number_of_elements:int,
                 activation_function:any, syntax_resp:list = []):
        self.number_of_neurons  = number_of_neurons
        self.number_of_elements = number_of_elements
        self.phi                = activation_function
        self.syntax_resp        = syntax_resp
        self.neurons            = []
        
        for i in range(number_of_neurons):
            self.neurons.append(neuron(number_of_elements, activation_function))
    
    def __repr__(self):
        representation = f'{self.number_of_neurons} neurons -> {self.number_of_elements} inputs'
        return representation
    
    def weight_atribute(self, weight_matrix:np.array):
        cols, rows = weight_matrix.shape
        assert cols == self.number_of_elements, 'Coluna não compatível com o número de elementos!'
        assert rows == self.number_of_neurons,  'Linha  não compatível com o número de neurônios!'
        
        for i in range(rows):
            self.neurons[i].weight_atribute(weight_matrix[i])
    
    def weight_randomize(self, maximum:int = 1, minimum:int=-1):
        for i in range(self.number_of_neurons):
            self.neurons[i].randomize_weight(maximum, minimum)
        
    def aplicate(self, data):
        assert len(data)+1 == self.number_of_elements, 'Dados incompatíveis com a camada!'
        
        results = []
        for  i in range(self.number_of_neurons):
            results.append(self.neurons[i].aplicate(data))
        return np.array(results)
    
    def syntax_aplicate(self, data):
        assert self.syntax_resp != [], 'Não existe resposta sintática armazenada'
        
        resp = self.aplicate(data)
        for numbers, value in self.syntax_resp:
            if all(numbers == resp):
                return value
        return resp


def training_layer (neuron_layer:layer, data:np.array, results:np.array,
                    eta = lambda ssns: 0.1, season_to_print = 1000, number_of_seasons = 100_000):
    
    rows, cols  = data.shape
    acum_error  = np.zeros(neuron_layer.number_of_neurons)
    acum_e_all  = np.zeros(neuron_layer.number_of_neurons)
    
    row_array   = np.arange(0, rows)
    for ssns in range(number_of_seasons):
        acum_error *= 0
        for row in row_array:
            phi_v       = neuron_layer.aplicate(data[row])
            error       = results[row] - phi_v
            acum_error += abs(error)
            acum_e_all += acum_error
            for i in range(neuron_layer.number_of_neurons):
                Xp  = np.r_[1, data[row]]
                wn1 = neuron_layer.neurons[i].weight + error[i] * Xp * eta(ssns)
                neuron_layer.neurons[i].weight_atribute( wn1 )
        
        np.random.shuffle(row_array)
        if (ssns%season_to_print == 0):
            space_string = ' '*len(f'{season_to_print}')
            print(f'\n________________________ Season: {ssns} ________________________')
            print(f'acumalate error in last {space_string} season: {acum_error}')
            print(f'acumalate error in last {season_to_print} season: {acum_e_all}')
            acum_e_all *= 0
            
        if sum(acum_error) == 0:
            print(f'\n________________________ Season: {ssns} ________________________')
            print(f'acumalate error in last season: {acum_error}')
            print(f'acumalate error in all {season_to_print}  season: {acum_e_all}')
            break
