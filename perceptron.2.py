import math
import numpy as np


alpha = 0.25
tolerancia = 0.01  # Margen de error aceptable

wl1 = np.random.random ([3, 2])
x = np.ones ([2,1])
y = np.ones ([2,1])
bl1 = np.ones ([3, 1])
a1 = np.ones ([3,1])
wl2 = np.random.random([2, 3])
bl2 = np.ones ([2,1])
a2 = np.ones([2,1])
zl1 = np.ones ([3,1])
zl2 = np.ones ([2,1])
error2 = np.ones ([2,1])
deltas1 = np.ones ([3, 1])
errorpesos2 = np.ones ([2, 3])
errorpesos1 = np.ones ([3, 2])


# Función sigmoide y su derivada
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def sigmoide_derivada(a):
    return a * (1 - a)


# Combinaciones de entrada y salida
datos = [
    (np.array([[0], [0]]), np.array([[0], [0]])),  # x = [0; 0], y = [0; 0]
    (np.array([[0], [1]]), np.array([[1], [0]])),  # x = [0; 1], y = [1; 0]
    (np.array([[1], [0]]), np.array([[1], [0]])),  # x = [1; 0], y = [1; 0]
    (np.array([[1], [1]]), np.array([[0], [1]])),  # x = [1; 1], y = [0; 1]
]
epoch = 0
# Entrenamiento para cada combinación
while epoch < 50000:

    for x, y in datos:
        a1 = (np.matmul(wl1, x)) + bl1
        zl1 = (sigmoide(a1))
        #print(f"Salida layer 1: {zl1}")

        a2 = (np.matmul(wl2, zl1)) + bl2
        zl2 = (sigmoide(a2))
        #print(f"Salida layer 2: {zl2}")

        '''Actualizar pesos 2'''
        error2 = (zl2 - y)
        #print(f"Error (delta z2): {error2}")

        # Verificar si el error está dentro de la tolerancia
        #if np.all(np.abs(error2) < tolerancia):
           # break

        '''salida layer 1 traspuesta'''
        zl1T = zl1.T
        # errorpesos2 = zl1T * error
        errorpesos2 = np.dot(error2, zl1T)

        wl2 = wl2 - alpha * errorpesos2
        #print(f"Pesos 2 actualizado: {wl2}")

        '''Actualizar pesos 1'''
        parte1 = np.dot(wl2.T, error2)
        parte2 = zl1 * (1 - zl1)

        deltas1 = parte1 * parte2
        #print(f"Deltas layer1: {deltas1}")

        '''x traspuesta'''
        xT = x.T
        errorpesos1 = np.dot(deltas1, xT)

        wl1 = wl1 - alpha * errorpesos1
        #print(f"Pesos 1 actualizado: {wl1}")

        epoch += 1

    # Mostrar resultados para cada combinación
   # print(f"Entrenamiento completado para x = {x.T[0]} e y = {y.T[0]} en {epoch} épocas")
   # print(f"Salida final (zl2): {zl2.T[0]}")
   # print("---------")


for x, y in datos:
    a1 = (np.matmul(wl1, x)) + bl1
    zl1 = (sigmoide(a1))
    #print(f"Salida layer 1: {zl1}")

    a2 = (np.matmul(wl2, zl1)) + bl2
    zl2 = (sigmoide(a2))
    #print(f"Salida layer 2: {zl2}")

    # Mostrar resultados para cada combinación
    print(f"Entrenamiento completado para x = {x.T[0]} e y = {y.T[0]} en {epoch} épocas")
    print(f"Salida final (zl2): {zl2.T[0]}")
    print("---------")