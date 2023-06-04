# Neural Network Models Repository

Repository contains various classification models implemented from scratch, showcasing my understanding of the underlying structure of neural networks. The models included are:

## 1. Logistic Regression

A logistic regression model implemented from scratch.

**Hyperparameters:**
- Learning rate: Controls the step size at each iteration during model training.
- Number of iterations: Determines the number of times the model updates its parameters using the training data.

## 2. Shallow Neural Network

A neural network with a single hidden layer.

**Hyperparameters:**
- Number of iterations: Specifies the number of times the model updates its parameters using the training data.
- Number of nodes in the single hidden layer: Defines the number of neurons in the hidden layer.
- Hidden layer activation function: Options include ReLU, sigmoid, or tanh.
- Output layer activation function: Can be either softmax for multiclass classification or sigmoid for binary classification.
- Learning rate: Controls the step size at each iteration during model training.
- Random seed: Provides reproducibility of the results.

## 3. Deep Neural Network

A deep neural network capable of handling multiple hidden layers.

**Hyperparameters:**
- Layer dimensions: Defines the structure of the network, where the first digit represents the number of features in the input data, the last digit denotes the number of output categories, and the numbers in between represent the number of neurons in each respective hidden layer.
- Number of iterations: Specifies the number of times the model updates its parameters using the training data.
- Learning rate: Controls the step size at each iteration during model training.
- Hidden layer activation: Applies to all hidden layers and can be set to relu, sigmoid, or tanh.
- Output layer activation: Can be sigmoid for binary classification or softmax for multiclass classification.
- Random seed: Provides reproducibility of the results.

## Potential Improvements:
- Addition of L1 and L2 regularization: Helps prevent overfitting and improves model generalization.
- Implementation of minibatches: Enables faster and more efficient training by processing data in smaller batches instead of the entire dataset at once.
- Improved initialization of random weights: Enhances the convergence speed and performance of the model.
- Handling exploding and vanishing gradients: Incorporate techniques like gradient clipping or using appropriate activation functions to address gradient stability issues.
- Explore techniques like dropout, batch normalization, or early stopping.
- Explore advanced architectures like residual connections, skip connections, or attention mechanisms to improve model performance.
- Extend the models to support regression tasks: Introduce a new cost function such as mean squared error (MSE) and adjust the output layer activation to match the requirements of regression.
