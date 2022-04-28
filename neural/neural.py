#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


deg = lambda v: 1 if v>0 else 0

tanh = lambda v: np.tanh(v/2)

sigmoid = lambda v: 1/(1 + np.exp(-v))

d_sigmoid = lambda v: sigmoid(v) * (1 - sigmoid(v))

def __signal__(x):
    if x >= 0:
        return '+'
    else:
        return '-'

 
class neuron:
    def __init__(self, number_of_elements:int, activation_function:any,
                 d_activation_function:any = None):
        self.number_of_elements = number_of_elements
        self.weight             = np.zeros(number_of_elements)
        self.dphi               = d_activation_function
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
    
    def induced_local_field(self, data):
        Xp = np.r_[1, data]
        v  = np.dot(Xp, self.weight.T)
        return v
        
    def aplicate(self, data):
        v = self.induced_local_field(data)
        return self.phi(v)  
     

class layer:
    def __init__(self, number_of_neurons:int, number_of_elements:int,
                 activation_function:any=sigmoid, d_activation_function:any = d_sigmoid,
                 syntax_resp:list = [], correct_val = 0.5):
        self.number_of_neurons  = number_of_neurons
        self.number_of_elements = number_of_elements
        self.phi                = activation_function
        self.dphi               = d_activation_function
        self.syntax_resp        = syntax_resp
        self.correct_val        = correct_val
        self.neurons            = []
        
        for i in range(number_of_neurons):
            self.neurons.append(neuron(number_of_elements, activation_function, d_activation_function))
    
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
    
    def induced_local_field(self, data:np.array):
        assert len(data)+1 == self.number_of_elements, 'Dados incompatíveis com a camada!'
        
        phi_v   = []
        dphi_v  = []
        for  i in range(self.number_of_neurons):
            v = self.neurons[i].induced_local_field(data)
            phi_v .append(self.neurons[i].phi(v))
            dphi_v.append(self.neurons[i].dphi(v))
            
        return  np.array(phi_v), np.array(dphi_v)
        
    def aplicate(self, data:np.array):
        assert len(data)+1 == self.number_of_elements, 'Dados incompatíveis com a camada!'
        
        results = []
        for  i in range(self.number_of_neurons):
            results.append(self.neurons[i].aplicate(data))
        return np.array(results)
    
    def absolute_aplicate(self, data):
        output = [ 1 if neuron_output >= self.correct_val else 0 
                 for neuron_output in self.aplicate(data)]
        return np.array(output)
    
    def syntax_aplicate(self, data):
        assert self.syntax_resp != [], 'Não existe resposta sintática armazenada'
        
        resp = self.aplicate(data)
        for numbers, value in self.syntax_resp:
            if all(numbers == resp):
                return value
        return resp


class mlp:
    def __init__(self, layer_list:list):
        self.output_layer   = layer_list[-1]
        self.correct_val    = self.output_layer.correct_val
        
        self.layer_list     = layer_list[:-1]
    
    def __repr__(self):
        resp = ''
        for layer in self.layer_list:
            resp += f'({layer.number_of_neurons})-->'
        resp+= f'({self.output_layer.number_of_neurons})'
        return resp
    
    def multi_answer(self, data:np.array):
        phi     = data.copy()
        phi_v   = []
        dphi_v  = []
        for layer in self.layer_list:
            phi, dphi = layer.induced_local_field(phi)
            phi_v .append(phi)
            dphi_v.append(dphi)
        
        phi,dphi = self.output_layer.induced_local_field(phi) 
        phi_v .append(phi)
        dphi_v.append(dphi)
        
        return phi_v, dphi_v
        
    def aplicate(self, data:np.array):
        values = data.copy()
        for layer in self.layer_list:
            values = layer.aplicate(values) 
        values = self.output_layer.aplicate(values)
        return values

    def absolute_aplicate(self, data:np.array):
        output = [ 1 if neuron_output >= self.correct_val else 0 
                 for neuron_output in self.aplicate(data)]
        return np.array(output)
      
    
    