import math
import numpy as np

alpha = 0.25
wl1 = np.ones ([3, 2])
x = np.ones ([2,1])
x[0, 0] = 0
x[1, 0] = 1
y = np.ones ([2,1])
bl1 = np.ones ([3, 1])
a1 = np.ones ([3,1])
wl2 = np.ones([2, 3])
bl2 = np.ones ([2,1])
a2 = np.ones([2,1])
zl1 = np.ones ([3,1])
zl2 = np.ones ([2,1])
error2 = np.ones ([2,1])
deltas1 = np.ones ([3, 1])
errorpesos2 = np.ones ([2, 3])
errorpesos1 = np.ones ([3, 2])

def sigmoide(z):
    out = 1 / (1 + np.exp(-z))
    return out

a1 = (np.matmul(wl1, x)) + bl1
zl1 = (sigmoide(a1))
print (f"Salida layer 1: {zl1}")

a2 = (np.matmul(wl2, zl1)) + bl2
zl2 = (sigmoide(a2))
print (f"Salida layer 2: {zl2}")


'''Actualizar pesos 2'''
error2 = (zl2 - y)
print (f"Error (delta z2): {error2}")
'''salida layer 1 traspuesta'''
zl1T = zl1.T
#errorpesos2 = zl1T * error
errorpesos2 = np.dot(error2,zl1T)

wl2 =  wl2 - alpha * errorpesos2
print (f"Pesos 2 actualizado: {wl2}")


'''Actualizar pesos 1'''
parte1 = np.dot(wl2.T, error2)
parte2 = zl1 * (1 - zl1)

deltas1 = parte1 * parte2
print (f"Deltas layer1: {deltas1}")

'''x traspuesta'''
xT = x.T
errorpesos1 = np.dot(deltas1,xT)

wl1 = wl1 - alpha * errorpesos1
print( f"Pesos 1 actualizado: {wl1}")