#%%
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils import resample

import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import numpy as np

#%% import dados
pasta  = 'data/'

inp_berries  = np.loadtxt(pasta + 'input_berries.txt')
out_berries = np.zeros([len(inp_berries), 3])
with open(pasta + 'output_berries.txt') as file:
    out_text = [i for i in file]
    for i,text in enumerate(out_text):
        text = text.replace('\n', '')
        if text == 'mertilo':
            out_berries[i,0] = 1 
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'bo')
        elif text == 'framboesa':
            out_berries[i,1] = 1
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'ro')
        else:
            out_berries[i,2] = 1 
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'gx')
plt.title('Classes')
plt.xlabel('Parâmetro A')
plt.ylabel('Parâmetro B')
plt.grid()
plt.show()
#%% treinamento
SR = [[np.array([1,0,0]), 'Mertilo'  ],
      [np.array([0,1,0]), 'Framboesa'],
      [np.array([0,0,1]), 'Açaí'     ]]

neuron_layer_deg    = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.deg    , syntax_resp=SR)
neuron_layer_sigm   = n.layer(number_of_neurons=3, number_of_elements=3, activation_function=n.sigmoid, syntax_resp=SR)

error_deg           = v.training_layer(neuron_layer_deg,  inp_berries, out_berries, eta = lambda a: 0.1)
error_sigm          = v.training_layer(neuron_layer_sigm, inp_berries, out_berries, eta = lambda a: 0.1)
#%% plot error
plt.plot(error_deg,  label='função degral', alpha = 0.5)
plt.plot(error_sigm, label='função sigmoidal',alpha = 0.5)
plt.title('Evolução do erro durante o treinamento')
plt.xlabel('iteração')
plt.ylabel('erro RMS')
plt.legend()
plt.show()

#%% Plot base de dados
for i,_ in enumerate(out_berries):
    if out_berries[i,0]:
        plt.plot(inp_berries[0],inp_berries[1], 'bo')
    

#%%