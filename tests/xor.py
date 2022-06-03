#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import neural.validation as v
import neural.neural as n
import matplotlib.pyplot as plt

a = n.mlp(2,2,1)
a.weight_randomize()

Xor_i = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

Xor_r = np.array([[0],
                  [1],
                  [1],
                  [0]])

error  = v.validate_mlp(a, Xor_i, Xor_r)

#%% Plot
plt.plot(error)
plt.xlabel('iteações')
plt.ylabel('error')
plt.title('Erro para XOR')
plt.grid()
plt.savefig('erro_xor.png')
plt.show()