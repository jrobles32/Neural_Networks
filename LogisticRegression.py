import numpy as np

from utils.sigmoid import sigmoid
from utils.loss_functions import loss_functions
from utils.logger import create_logger


class LogisticRegression:
    """
    Creates a loglistic regression object focused on binary classification. Model is
    optimized by using binary cross entropy loss. 

    Parameters
    ----------
    iterations: int
        Number of gradient steps to take.
    lr: float
        Size of step taken in each iteration. Size is relative to gradient of the step.
    cache_cost: int
        Number of iterations that need to occur before cost values are stored.
    verbose: bool
        If true prints out the cost every time they get stored in the cache.

    Attributes
    ----------
    interations: int
        Desired number of iterations for the model.
    lr: float
        Desired learning rate of the model.
    w: numpy.ndarray
        Learned parameters for the logistic regression. One for each feature.
    b: float
        Learned constant for the logistic regression.
    cache_cost: int
        After how many iterations costs are being stored.
    costs: list
        A list storing the results of the cost function. Called based on cache_cost iterations.
    verbose: bool
        Whether or not costs values are being printed in terminal.

    Methods
    -------
    fit
        Fits the passed features and target.
    predict
        Predicts the class for a set of features.
    """
    def __init__(self, iterations=1000, lr=0.001, cache_cost=500, verbose=True):
        self.lr = lr
        self.iterations = iterations
        self.cache_cost = cache_cost
        self.verbose = verbose

        self.w = None
        self.b = 0

        self.loss_function = loss_functions.binary_cross_entropy
        self.costs = []

        self.logger = create_logger('Logistic')

    
    def fit(self, X, Y):
        """
        Fitting the inputted features and target variable to create the logistic regression
        model.

        Parameters
        ----------
        X: numpy.ndarray
            Features to train the model. Shape of (number of features, number of samples)
        Y: numpy.ndarray
            Target labels relative to the inputted features. Shape of (1, number of samples )
        """
        assert X.shape[1] == Y.shape[1], "X or Y is not being defined correctly. Need shape[1] to be the number of samples"

        # initializing weights as zero and determining the number of samples
        self.w = np.zeros((X.shape[0], 1))
        m = X.shape[1]

        for i in range(1, self.iterations + 1):
            # forward prop
            Z = np.dot(self.w.T, X) + self.b
            A = sigmoid.function(Z)
            
            # backward prop to find gradients
            dz = A - Y
            dw = (1/m) * np.dot(X, dz.T)
            db = (1/m) * np.sum(dz)

            # applying gradients to weights
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
   
            # calculating cost and storing the value
            if i % self.cache_cost == 0:
                cost = self.loss_function(A, Y, m)
                self.costs.append(cost)

                if self.verbose:
                    self.logger.info(f'Cost after iteration {i}: {round(float(np.squeeze(cost)), 5)}')
    

    def predict(self, X):
        """
        Helps determine the class label for a set of features. Label is based on the weights 
        and constant found during fit of the model.

        Parameters
        ----------
        X: numpy.ndarray
            Set of features to predict label. Shape of (number of features, number of samples).
            Number of features should be the same as the number used to fit the model.
        
        Returns
        -------
        y_prediction: numpy.ndarray
            Predict class label for each sample. Shape of (1, number of samples).
        """
        # applying forward propagation step to inputted features
        A = sigmoid.function(np.dot(self.w.T, X) + self.b)

        # determining class label based on forward propagation step
        y_prediction = np.where(A >= 0.5, 1, 0)
        
        return y_prediction
