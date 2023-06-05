import numpy as np

class relu:
    @staticmethod
    def function(z):
        #a = z * (z > 0)
        a = np.maximum(0, z)
        return a

    @staticmethod
    def derivative(z):
        #d = (z > 0) * 1
        d = np.where(z > 0, 1, 0)
        return d