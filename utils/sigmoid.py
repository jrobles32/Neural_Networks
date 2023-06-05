import numpy as np

class sigmoid:
    @staticmethod
    def function(z):
        a = 1 / (1 + np.exp(-z))
        return a

    @staticmethod
    def derivative(z):
        a = sigmoid.function(z)
        d = a * (1 - a)
        return d