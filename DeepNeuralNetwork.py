import numpy as np

from utils.tanh import tanh
from utils.sigmoid import sigmoid
from utils.relu import relu
from utils.softmax import softmax
from utils.loss_functions import loss_functions
from utils.logger import create_logger


class DeepNeuralNetwork:
    """
    Creates a deep neural network object focused on classification. 

    Parameters
    ----------
    layer_dims: list
        First digit of list should be the number of features. Subsequent digits represent the number of neurons
        in respective layer. Final digit represents the number of categories.
    iterations: int
        Number of gradient steps to take.
    lr: float
        Learning rate for each gradient step.
    hidden_activation: str
        Nonlinear activation that should be used in the hidden layers. Available options are sigmoid,
        relu, and tanh.
    output_activation: str
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
    layer_dims: list
        Dimensions of each layer in neural network.
    num_layers: int
        Total number of layers in network.
    interations: int
        Total number of iterations for the model.
    interior_activation: func
        Nonlinear activation function of the hidden layer.
    interior_derivative: func
        Derivative of the interior_activation. Important for backpropagation.
    output_activation: func
        Nonlinear activation function of the output layer.
    output_derivative: func
        Derivative of the output_activation. Important for backpropagation.
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
    costs: list
        A list storing the results of the cost function. Called based on cache_cost iterations.
    cache_cost: int
        After how many iterations costs are being stored.
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
            layer_dims,
            iterations, 
            lr, 
            hidden_activation, 
            output_activation, 
            seed, 
            cache_cost=500, 
            verbose=True
    ):
        np.random.seed(seed)
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)

        self.iterations = iterations
        self.lr = lr
        self.cache_cost = cache_cost
        self.verbose = verbose

        self.parameters = self._initialize_weights()

        self.available_activations = {
            'sigmoid': [sigmoid.function, sigmoid.derivative],
            'relu': [relu.function, relu.derivative],
            'tanh': [tanh.function, tanh.derivative],
            'softmax': [softmax.function, None]
        }

        self.interior_activation, self.interior_derivative = self.available_activations.get(hidden_activation)
        self.output_activation, self.output_derivative = self.available_activations.get(output_activation)
        self.loss_func = loss_functions.binary_cross_entropy if output_activation == 'sigmoid' else loss_functions.cross_entropy

        self.costs = []
        self.logger = create_logger('DeepNeuralNetwork')


    def _initialize_weights(self):
        """
        Creates randomly initiated weights for each layer. Also creating the required constants. 
        Important step for forward propagation.

        Returns
        -------
        parameters: dict
            The randomly initiated weights for each layer in deep neural network.
        """
        parameters = {}

        for l in range(1, self.num_layers):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

        return parameters


    def _forward_propagation(self, X):
        """
        Parameters
        ----------
        X: numpy.ndarray
            Set of features to predict label. Shape of (number of features, number of samples).
        
        Returns
        -------
        cache[f'A{L}']: numpy.ndarray
            Final result of the forward propagation. The prediction values (y-hat).
        cache: dict
            Contains all the intermediary results of the forward progation. Important for backward propagation.
        """
        # creating a dictionary to store the results of the forward propagation
        # determining key of last layer
        cache = {'A0':X}
        L = self.num_layers - 1

        # finding the linear and activation values for each layer
        for l in range(1, L):
            cache[f'Z{l}'] = np.dot(self.parameters[f'W{l}'], cache[f'A{l-1}']) + self.parameters[f'b{l}']
            cache[f'A{l}'] = self.interior_activation(cache[f'Z{l}'])
        
        # finding the linear and activation value for the last layer
        cache[f'Z{L}'] = np.dot(self.parameters[f'W{L}'], cache[f'A{L-1}']) + self.parameters[f'b{L}']
        cache[f'A{L}'] = self.output_activation(cache[f'Z{L}'])
        
        return cache[f'A{L}'], cache
    

    def _backward_propagation(self, y, m, cache):
        """
        Peforms the backward propagation of the model fitting process.

        Parameters
        ----------
        y: numpy.ndarray
            Actual label for each sample.
        m: int
            Total number of samples.
        cache: dict
            Dictionary containing the A and Z values for each layer. Created in the forward propagation step.
        """
        # getting the highest key for a layer and creating dictionary to store gradients
        L = self.num_layers - 1
        gradients = {}

        # reshaping y to be compatible with last layer
        y = y.reshape(cache[f'A{L}'].shape)

        # finding the last layers dZ based on activation in output layer
        if self.output_activation == sigmoid.function:
            dAL = -(np.divide(y, cache[f'A{L}']) - np.divide(1 - y, 1 - cache[f'A{L}']))
            dZ = dAL * self.output_derivative(cache[f'Z{L}'])

        else:
            dZ = cache[f'A{L}'] - y

        # storing the gradients for the last layer
        gradients[f'dW{L}'] = (1/m) * np.dot(dZ, cache[f'A{L-1}'].T)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # finding the gradients for all other layers
        for l in reversed(range(1, L)):
            dA_prev = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA_prev * self.interior_derivative(cache[f'Z{l}'])

            gradients[f'dW{l}'] = (1/m) * np.dot(dZ, cache[f'A{l-1}'].T)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # updating the parameters
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= gradients[f'dW{l}'] * self.lr
            self.parameters[f'b{l}'] -= gradients[f'db{l}'] * self.lr


    def fit(self, X, y):
        """
        Fitting inputted features and target variable to create the deep neural network
        model.

        Parameters
        ----------
        X: numpy.ndarray
            Features to train the model. Shape of (number of features, number of samples)
        Y: numpy.ndarray
            Target labels relative to the inputted features. Shape of (number of classes, number of samples)
        """
        assert X.shape[1] == y.shape[1], "X or Y is not being defined correctly. Need shape[1] to be the number of samples"

        # determining the total number of samples
        m = X.shape[1]

        for i in range(1, self.iterations+1):
            # performing forward and backward propagation
            y_hat, cache = self._forward_propagation(X)
            self._backward_propagation(y, m, cache)

            # calculating cost and storing the value
            if i % self.cache_cost == 0:
                cost = self.loss_func(y_hat, y, m)
                cost = round(float(np.squeeze(cost)), 5)
                self.costs.append(cost)

                if self.verbose:
                    self.logger.info(f'Cost after {i} iterations: {cost}')


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
        # applying forward propagation
        y_hat, _ = self._forward_propagation(X)

        # determing class label based on activation used in output layer
        if self.output_activation == softmax.function:
            return np.argmax(y_hat, axis=0)

        else:
            return np.where(y_hat >= 0.5, 1, 0)
