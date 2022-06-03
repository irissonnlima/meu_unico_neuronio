#%%
import matplotlib.pyplot as plt
import neural.validation as v
import neural.neural as n
import numpy as np

np.random.seed(666)
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
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'kx')
        elif text == 'framboesa':
            out_berries[i,1] = 1
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'kd')
        else:
            out_berries[i,2] = 1 
            plt.plot(inp_berries[i,0],inp_berries[i,1], 'kv')
plt.title('Classes')
plt.xlabel('Parâmetro A')
plt.ylabel('Parâmetro B')
plt.grid()
plt.savefig('classes_berry.png')
plt.show()

#%% treinamento
neuron_layer_deg    = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.deg    )
neuron_layer_sigm   = n.layer(number_of_neurons=3, number_of_elements=2, activation_function=n.sigmoid)

#%%
error_deg           = v.training_layer(neuron_layer_deg,  inp_berries, out_berries, eta = lambda a: 0.1)
error_sigm          = v.training_layer(neuron_layer_sigm, inp_berries, out_berries, eta = lambda a: 0.1)

#%% plot error
plt.plot(error_deg,  label='função degrau', alpha = 0.5)
plt.plot(error_sigm, label='função sigmoidal',alpha = 0.5)
plt.title('Evolução do erro durante o treinamento')
plt.xlabel('iteração')
plt.ylabel('erro RMS')
plt.legend()
plt.savefig('berries_deg_sig.png')
plt.show()

#%% Plot base de dados
for i,_ in enumerate(out_berries):
    if out_berries[i,0]:
        plt.plot(inp_berries[0],inp_berries[1], 'bo')
    

#%%Bootstrap
np.random.seed(666)
avg, std,_ = v.bootstrap_layer(inp_berries, out_berries,
                               reset_function    = lambda : neuron_layer_sigm.weight_randomize(),
                               training_function = lambda inp, out: v.training_layer(neuron_layer_sigm, inp, out),
                               aplicate_function = lambda val: neuron_layer_sigm.absolute_aplicate(val))
#RMS in last season: 1.50e-05
#~~> erro bootstrap: 0.25
