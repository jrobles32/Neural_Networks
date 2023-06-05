import numpy as np

class softmax:
    @staticmethod
    def function(z):
        expz = np.exp(z - np.max(z, axis=0, keepdims=True))
        a = expz / np.sum(expz, axis=0, keepdims=True)
        return a