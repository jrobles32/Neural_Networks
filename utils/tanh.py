import numpy as np

class tanh:
    @staticmethod
    def function(z):
        a = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return a

    @staticmethod
    def derivative(z):
        a = tanh.function(z)
        d = 1 - np.power(a, 2)
        return d