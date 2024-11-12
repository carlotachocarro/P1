import math
import numpy as np

alpha = 0.25
wl1 = np.ones ([3, 2])
x = np.array ([0, 1])
y = np.array ([0, 1])
bl1 = np.ones ([3])
a1 = np.ones ([3])
wl2 = np.ones([2, 3])
bl2 = np.ones ([2])
a2 = np.ones([2])
zl1 = np.array ([3])
zl2 = np.array ([2])
error = np.ones ([2])
errorpesos2 = np.ones ([2, 3])

def sigmoide(z):
    out = 1 / (1 + math.e**(-z))
    return out

a1 = (np.dot(wl1, x)) + bl1
zl1 = (sigmoide(a1))
print (f"Salida layer 1: {zl1}")

a2 = (np.dot(wl2, zl1)) + bl2
zl2 = (sigmoide(a2))
print (f"Salida layer 2: {zl2}")

error = (zl2 - y)
print (f"Error (delta z): {error}")

'''Actualizar pesos 2'''
'''salida layer 1 traspuesta'''
zl1T = zl1.T
errorpesos2 = zl1T * error
wl2 = wl2 - alpha * errorpesos2
print (f"Pesos 2 actualizado: {wl2}")

