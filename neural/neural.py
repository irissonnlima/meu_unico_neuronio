#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

#%% Funções de ativação e derivadas
def deg(v):
    val = []
    for i in v[0,:]:
        if i > 0:
            val.append(1)
        else:
            val.append(0)
    return np.array([val])
#deg         = lambda v: 1 if v>0 else 0

tanh        = lambda v: np.tanh(v/2)

sigmoid     = lambda v: 1/(1 + np.exp(-v))
d_sigmoid   = lambda v: sigmoid(v) * (1 - sigmoid(v))

#%% Elementos neuronios
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
        self.number_of_elements = number_of_elements+1
        
        self.phi                = activation_function
        self.dphi               = d_activation_function
        
        self.syntax_resp        = syntax_resp
        self.correct_val        = correct_val
        
        self.weights            = np.zeros([self.number_of_neurons, self.number_of_elements])
    
    def __repr__(self):
        representation = f'{self.number_of_neurons} neurons -> {self.number_of_elements-1} inputs'
        return representation
    
    def weight_atribute(self, weight_matrix:np.array):
        cols, rows = weight_matrix.shape
        assert cols == self.number_of_elements, 'Coluna não compatível com o número de elementos!'
        assert rows == self.number_of_neurons,  'Linha  não compatível com o número de neurônios!'
        self.weights = weight_matrix
    
    def weight_randomize(self, maximum:int = 1, minimum:int=0):
        k = (maximum-minimum)*maximum
        self.weights = k*np.random.random([self.number_of_neurons, self.number_of_elements])

    def aplicate(self, data:np.array):
        if len(data.shape) == 1:
            data = np.array([data])
        r,c = data.shape
        assert (r, c+1) == (1, self.number_of_elements), 'Dados incompatíveis com a camada!'
        data = np.c_[[1],data]
        resp = self.weights.dot(data.T)
        resp = self.phi(resp.T)
        return resp
        
    def absolute_aplicate(self, data):
        output = [ 1 if neuron_output >= self.correct_val else 0 
                 for neuron_output in self.aplicate(data)[0]]
        return np.array([output])
    
    def save_weight(self, name:str):
        np.savetxt(f'{name}.weights',self.weights)

def reconstruct_layer(file_name:str, activation_function:any=sigmoid,
                      d_activation_function:any = d_sigmoid, syntax_resp:list = [],
                      correct_val = 0.5):
    weight = np.loadtxt(f'{file_name}.weights')
    number_of_neurons, number_of_elements = weight.shape
    l = layer(number_of_neurons, number_of_elements, activation_function,
              d_activation_function, syntax_resp, correct_val)
    
    for i,neuron in enumerate(l.neurons):
        neuron.weight = weight[i]
    return l

#%% Testes    
class mlp:
    def __init__(self, input_len=2, hidden_len=2, out_len=2, activation_function = sigmoid, d_activation_function = d_sigmoid):
        self.input_len              = input_len
        self.hidden_len             = hidden_len
        self.out_len                = out_len
        self.activation_function    = activation_function
        self.d_activation_function  = d_activation_function
        
        self.hidden                 = np.zeros([self.hidden_len, self.input_len+1])
        self.output                 = np.zeros([self.out_len   , self.hidden_len+1])
    
    def weight_randomize(self):
        self.hidden = np.random.random([self.hidden_len, self.input_len+1])
        self.output = np.random.random([self.out_len  , self.hidden_len+1])
    
    def aplicate(self, data:np.array):
        if len(data.shape) == 1:
            data = np.array([data])
        Xp      = np.c_[[1], data].T
        # aplicação da entrada
        vi      = self.hidden.dot(Xp)
        phi_i   = self.activation_function(vi)
        
        #camada de saída
        vo      = self.output.dot(np.r_[[[1]], phi_i])
        phi_o   = self.activation_function(vo)
        return vi, vo, phi_i, phi_o
       
