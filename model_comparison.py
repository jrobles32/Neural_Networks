import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogitRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from NeuralNetwork import NeuralNetwork
from DeepNeuralNetwork import DeepNeuralNetwork
from LogisticRegression import LogisticRegression

output_type = 'sigmoid'

binary_path = r'D:\Jupyter_Conversion\datasets\telecom_churn_clean.csv'.replace('\\', '/')
binary_target = 'churn'

multiclass_path = r'D:\Jupyter_Conversion\datasets\mobile_price_classification.csv'.replace('\\', '/')
multiclass_target = 'price_range'

def base_line(x_train, x_test, y_train, y_test):
    sklearn_model = LogitRegression(max_iter=5000)
    sklearn_model.fit(x_train, y_train)

    score_val_train = sklearn_model.score(x_train, y_train)
    score_val = sklearn_model.score(x_test, y_test)
    print('Baseline Train Score:', score_val_train)
    print('Baseline Test Score:', score_val)
    return 


def binary_classification(path, target, train_size=0.8):
    test_data = pd.read_csv(path)

    x_data = test_data.drop(labels=[target], axis=1).values
    y_data = test_data[target].values

    ss = StandardScaler()
    x_data = ss.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=10)
    base_line(x_train, x_test, y_train, y_test)

    print('x-train shape', x_train.T.shape)
    print('y-train shape', y_train.T.reshape(1, -1).shape)
    return x_train.T, x_test.T, y_train.T.reshape(1, -1), y_test.T.reshape(1, -1)


def multi_classification(path, target, train_size=0.8):
    test_data = pd.read_csv(path)

    x_data = test_data.drop(labels=[target], axis=1).values
    y_data = test_data[target].values

    ss = StandardScaler()
    x_data = ss.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=10)
    base_line(x_train, x_test, y_train, y_test)

    oh = OneHotEncoder()
    y_train = oh.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = oh.fit_transform(y_test.reshape(-1, 1)).toarray()

    print('x-train shape', x_train.T.shape)
    print('y-train shape', y_train.T.shape)
    return x_train.T, x_test.T, y_train.T, y_test.T


def score(y, ypred, type_class='sigmoid'):
    if type_class == 'sigmoid':
        return accuracy_score(y.flatten(), ypred.flatten())
    
    elif type_class == 'softmax':
        target= np.argmax(y, axis=0)
        return accuracy_score(target, ypred)
    

if output_type == 'sigmoid':
    x_train_t, x_test_t, y_train_t, y_test_t = binary_classification(binary_path, binary_target)

    third_model = LogisticRegression(iterations=5000, lr=0.01)
    third_model.fit(x_train_t, y_train_t)

    y_pred_train_logit = third_model.predict(x_train_t)
    y_pred_logit = third_model.predict(x_test_t)

    print("Logit train:", score(y_train_t, y_pred_train_logit, type_class=output_type))
    print("Logit test:", score(y_test_t, y_pred_logit, type_class=output_type))

else:
    x_train_t, x_test_t, y_train_t, y_test_t = multi_classification(multiclass_path, multiclass_target)

model = NeuralNetwork(iterations=5000, lr=0.01, nodes=50, hidden_layer='relu', output_layer=output_type, seed=112)
model.fit(x_train_t, y_train_t)

y_pred_train = model.predict(x_train_t)
y_pred = model.predict(x_test_t)

print("NN train:", score(y_train_t, y_pred_train, type_class=output_type))
print("NN test:", score(y_test_t, y_pred, type_class=output_type))

second_model = DeepNeuralNetwork(layer_dims=[x_train_t.shape[0], 10, 10, y_train_t.shape[0]], iterations=5000, hidden_activation='relu', output_activation=output_type, lr=0.1, seed=112)
second_model.fit(x_train_t, y_train_t)

y_pred_train_deep = second_model.predict(x_train_t)
y_pred_deep = second_model.predict(x_test_t)

print("DNN train:", score(y_train_t, y_pred_train_deep, type_class=output_type))
print("DNN test:", score(y_test_t, y_pred_deep, type_class=output_type))
