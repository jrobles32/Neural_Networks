import numpy as np

class loss_functions:
    @staticmethod
    def binary_cross_entropy(y_hat, Y, m):
        loss = -(1/m) * (np.dot(Y, np.log(y_hat).T) + np.dot((1-Y), np.log(1-y_hat).T))
        return loss
    
    @staticmethod
    def cross_entropy(y_hat, Y, m):
        loss = (1/m) * -np.sum(Y * np.log(y_hat + 1e-8))
        return loss
    