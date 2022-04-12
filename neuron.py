#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def __signal__(x):
    if x >= 0:
        return '+'
    else:
        return '-'
    
    
class neuron:
    def __init__(self, number_of_elements:int):
        self.number_of_elements = number_of_elements
        self.weight             = np.zeros(number_of_elements)

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
        return v
        
    def __repr__(self):
        representation = '+' + '-'*14 + '+\n'
        for i in range(self.number_of_elements):
            representation += f'|w{i} = {__signal__(self.weight[i])}{abs(self.weight[i]):.2e}|\n'
        representation += '+' + '-'*14 + '+\n'
        return representation


def training_function(qtde:int, activation_function:any, data:np.array,
                      results:np.array, number_of_seasons = 70_000):
    
    rows, cols  = data.shape
    n = [neuron(cols+1) for i in range(qtde)]
    
    for ssns in range(number_of_seasons):
        print(f'--------------iteração: {ssns}------------------')
        for row in range(rows):
            
            v       = np.array([n[i].aplicate(data[row]) for i in range(qtde)])
            error   = results[row] - activation_function(v)
            [n[i].weight_atribute( n[i].weight + error[i]*np.r_[1, data[row]]*0.1 ) for i in range(qtde)]
            print(f'error = {error}')
    return n

#%%]
def deg(x):
    resp = np.zeros(x.shape)
    for i in range(len(x)):
        if  x[i]>0:
            resp[i] = 1
        else:
            resp[i] = 0
    return resp
  
Input  = np.loadtxt('input.txt')
Output = np.zeros([len(Input), 3])
with open('output.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'laranja':
            Output[i,2] = 1 
        elif text == 'maca':
            Output[i,1] = 1
        elif text == 'robson':
            Output[i,0] = 1
            
for i in range(len(Output)):
    if Output[i,2] == 1:
        plt.plot(Input[i,0], Input[i,1], 'yx')
    elif Output[i,1] == 1:
        plt.plot(Input[i,0], Input[i,1], 'ro')
    elif Output[i,0] == 1:
        plt.plot(Input[i,0], Input[i,1], 'bo')
plt.show()
    
    
#%%
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    