import numpy as np

from utils.tanh import tanh
from utils.sigmoid import sigmoid
from utils.relu import relu
from utils.softmax import softmax
from utils.loss_functions import loss_functions
from utils.logger import create_logger


class NeuralNetwork:
    """
    Creates a neural network object focused on classification. 

    Parameters
    ----------
    iterations: int
        Number of gradient steps to take.
    lr: float
        Size of step taken in each iteration. Size is relative to gradient of the step.
    nodes: int
        Number of nodes in hidden layer.
    hidden_layer: str
        Nonlinear activation that should be used in the hidden layer. Available options are sigmoid,
        relu, and tanh.
    output_layer: str
        Nonlinear activation that should be used in the output layer. Available options are sigmoid 
        and softmax. Softmax should be used for multiclass classification.
    cache_cost: int
        Number of iterations that need to occur before cost values are stored.
    verbose: bool
        If true prints out the cost every time they get stored in the cache.
    seed: int
        Desired random state for the created weights.

    Attributes
    ----------
    interations: int
        Total number of iterations for the model.
    nodes: int
        Total number of nodes in hidden layer.
    hidden_activation: func
        Nonlinear activation function of the hidden layer.
    hidden_derivative: func
        Derivative of the activation_func1. Important for backpropagation.
    output_activation: func
        Nonlinear activation function of the output layer.
    available_activations: dict
        A dictionary containing the available activations as keys. Values being the function for 
        the activation, and its derivative form. 
    lr: float
        Desired learning rate of the model.
    parameters: dict
        Learned parameters for the neural network given a set of features. Includes the weights and
        constants for hidden and output layer.
    loss_func: func
        Function used as the loss function.
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
    def __init__(
            self, 
            iterations=2000, 
            nodes=50, 
            hidden_layer='relu', 
            output_layer='sigmoid', 
            lr=0.01, 
            cache_cost=500,
            verbose=True,
            seed=10
    ):
        np.random.seed(seed)
        self.lr = lr
        self.iterations = iterations
        self.nodes = nodes
        self.cache_cost = cache_cost
        self.verbose = verbose

        self.available_activations = {
            'sigmoid': [sigmoid.function, sigmoid.derivative],
            'relu': [relu.function, relu.derivative],
            'tanh': [tanh.function, tanh.derivative],
            'softmax': [softmax.function]
        }

        self.parameters = {
            'W1': None,
            'B1': None,
            'W2': None,
            'B2': None
        }

        self.hidden_activation, self.hidden_derivative = self.available_activations.get(hidden_layer)
        self.output_activation = self.available_activations.get(output_layer)[0]
        self.loss_func = loss_functions.binary_cross_entropy if output_layer == 'sigmoid' else loss_functions.cross_entropy

        self.costs = []

        self.logger = create_logger('NeuralNetwork')
    

    def fit(self, X, Y):
        """
        Fitting inputted features and target variable to create the shallow neural network
        model.

        Parameters
        ----------
        X: numpy.ndarray
            Features to train the model. Shape of (number of features, number of samples)
        Y: numpy.ndarray
            Target labels relative to the inputted features. Shape of (number of classes, number of samples)
        """
        assert X.shape[1] == Y.shape[1], "X or Y is not being defined correctly. Need shape[1] to be the number of samples"

        # determining the number of samples and initializing parameter weights
        m = X.shape[1]
        self.parameters['W1'] = np.random.randn(self.nodes, X.shape[0]) * 0.01
        self.parameters['B1'] = np.zeros((self.nodes, 1))
        self.parameters['W2'] = np.random.randn(Y.shape[0], self.nodes) * 0.01
        self.parameters['B2'] = np.zeros((Y.shape[0], 1))

        for i in range(1, self.iterations + 1):
            # forward prop
            Z1 = np.dot(self.parameters['W1'], X) + self.parameters['B1']
            A1 = self.hidden_activation(Z1)
            Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['B2']
            A2 = self.output_activation(Z2)
            
            # backward prop to find gradients
            dz2 = A2 - Y
            dw2 = (1/m) * np.dot(dz2, A1.T)
            db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
            dz1 = np.multiply(np.dot(self.parameters['W2'].T, dz2), (self.hidden_derivative(A1)))
            dw1 = (1/m) * np.dot(dz1, X.T)
            db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

            # applying gradients to weights
            self.parameters['W2'] -= self.lr * dw2
            self.parameters['B2'] -= self.lr * db2
            self.parameters['W1'] -= self.lr * dw1
            self.parameters['B1'] -= self.lr * db1
   
            # calculating cost and storing the value
            if i % self.cache_cost == 0:
                cost = self.loss_func(A2, Y, m)
                self.costs.append(cost)

                if self.verbose:
                    self.logger.info(f'Cost after {i} iterations: {round(float(np.squeeze(cost)), 5)}')
    

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
            Predicted class label for each sample.
        """
        # applying forward propagation step to inputted features
        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['B1']
        A1 = self.hidden_activation(Z1)
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['B2']
        A2 = self.output_activation(Z2)

        # determining class label based on forward propagation step
        if self.output_activation == softmax.function:
            return np.argmax(A2, axis=0)

        else:
            return np.where(A2 >= 0.5, 1, 0)
