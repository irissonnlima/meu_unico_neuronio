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
    def __init__(self, number_of_elements:int, activation_function:any):
        self.number_of_elements = number_of_elements
        self.weight             = np.zeros(number_of_elements)
        self.phi                = activation_function

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
        
    def __repr__(self):
        representation = '+' + '-'*14 + '+\n'
        for i in range(self.number_of_elements):
            representation += f'|w{i} = {__signal__(self.weight[i])}{abs(self.weight[i]):.2e}|\n'
        representation += '+' + '-'*14 + '+\n'
        return representation

def training_function(qtde:int, activation_function:any, data:np.array,
                      results:np.array,eta = lambda ssns: 0.1,
                      season_to_print = 1000, number_of_seasons = 100_000):
    
    rows, cols  = data.shape
    n           = [neuron(cols+1, activation_function) for i in range(qtde)]
    acum_error  = np.zeros(len(n))
    acum_e_all  = np.zeros(len(n))
    
    for ssns in range(number_of_seasons):
        acum_error *= 0
        for row in range(rows):
            phi_v       = np.array([n[i].aplicate(data[row]) for i in range(qtde)])
            error       = results[row] - phi_v
            acum_error += abs(error)
            acum_e_all += acum_error
            for i in range(qtde):
                Xp  = np.r_[1, data[row]]
                wn1 = n[i].weight + error[i] * Xp * eta(ssns)
                n[i].weight_atribute( wn1 )

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
    return n

#%%]
def deg(x):
    if  x>0:
        return 1
    else:
        return 0
  
Input  = np.loadtxt('input.txt')
Output = np.zeros([len(Input), 2])
with open('output.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'laranja':
            Output[i,0] = 1 
        elif text == 'maca':
            Output[i,1] = 1

for i in range(len(Output)):
    if Output[i,0] == 1:
        plt.plot(Input[i,0], Input[i,1], 'yx')
    elif Output[i,1] == 1:
        plt.plot(Input[i,0], Input[i,1], 'ro')

plt.show()
    
    
#%%
def f(ssns):
    x0 = 5.13e-2
    xf = 2.3026
    x  = (xf-x0)/(100_000) * ssns + x0
    return np.exp(-x)

n = training_function(2, deg, Input, Output, eta=f, number_of_seasons=100_000) 

    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    