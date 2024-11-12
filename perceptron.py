import math

class perceptron ():
    def __init__ (self):
        self.w = [1, 1, 1]
        self.x = [0, 1, 0]
        self.b = 1

    def sumatorio(self):
        sum = 0
        for i in range(len(self.w)):
            sum += self.w[i] * self.x[i]
        return sum + self.b

    def sigmoide (self, z):
        out = 1 / (1 + math.exp(-z))
        return out

perceptron = perceptron()

z = perceptron.sumatorio()
print (f"Sumatorio de (z): {z}")

sig = perceptron.sigmoide(z)
print (f"Sigmoide de (z): {sig}")
