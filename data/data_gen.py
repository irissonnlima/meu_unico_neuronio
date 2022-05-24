import numpy as np
import random as r

r.seed(2022)

fi = lambda m, M, p: (M-m)*p + m

X1 = lambda p: fi(3.45, 7.48, p)
Y1 = lambda p: fi(5.95, 8.60, p)

X2 = lambda p: fi(8.00, 20.00, p)
Y2 = lambda p: fi(9.00, 15.00, p)

X3 = lambda p: fi(19.00, 24.00, p)
Y3 = lambda p: fi(5.95, 7.00, p)

tam  = 150
matrix = np.zeros([tam, 2])
resp   = []
for i in range(tam):
    if i<tam/3:
        matrix[i,0] = X1(r.random())
        matrix[i,1] = Y1(r.random())
        resp.append('morango')
    elif i<2*tam/3:
        matrix[i,0] = X2(r.random())
        matrix[i,1] = Y2(r.random())
        resp.append('baunilha')
    else:
        matrix[i,0] = X3(r.random())
        matrix[i,1] = Y3(r.random())
        resp.append('chocolate')

np.savetxt('input_napolitano.txt', matrix, delimiter= '\t', fmt='%.4f')
with open("output_napolitano.txt",'a') as file:
    for _,name in enumerate(resp):
        file.write(name + '\n')