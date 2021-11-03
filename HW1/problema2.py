import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import warnings

np.random.seed(1)


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z 

def relu(Z):
    return np.maximum(0, Z), Z

def linear_activation_forward(A_prev, W, b, activation="relu"):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "none":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = Z
        activation_cache = Z 
    elif activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
                
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, last_activation):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)
        
    
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation=last_activation)
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, formula = 'MSE'):
    m = Y.shape[1]
    if formula == 'MSE':    
        cost = np.sum((Y-AL)**2)
    else:
        """When K = 2, as in this case, the cost function can be as the one used here. When K > 2, the loss would have to be computed individually
        for each class and then comput the sumation"""
        cost = -float(np.dot(np.log(AL),Y.T) + np.dot(np.log(1-AL),(1-Y).T))/float(m)
    cost = np.squeeze(cost)
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.matmul(dZ,A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.matmul(W.T, dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        dZ = np.copy(dA)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    warnings.simplefilter("error")
    try:
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of the loss function
    except RuntimeWarning as e:
        exit()
    

    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters



def L_layer_model(X_test, Y_test, X, Y, layers_dims, last_activation, learning_rate = 0.1, num_iterations = 3000, print_cost=True):
  
    np.random.seed(1)
    costs = []  
    costs_test = []                       # keep track of cost
    rms_costs = [] 
    parameters = initialize_parameters_deep(layers_dims)
    lastAL = None
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        lastAL = AL
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            AL, caches = L_model_forward(X_test, parameters)
            cost_test = compute_cost(AL, Y_test)
            costs_test.append(cost_test)
        costs.append(cost)
    fig = plt.figure(figsize=(15,15))
    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.plot(np.squeeze(costs_test))
    plt.ylabel('costs_test')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))

    fig.add_subplot(rows, columns, 2)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))

    plt.show()
    return parameters

def predict(X,Y,pretrained_weights):
    AL, caches = L_model_forward(X, parameters)
    cost = compute_cost(AL, Y)
    rms = computeRMS(AL, Y)
    print ("Cost after prediction test data %f" %(rms))

def computeRMS(AL,Y):
    rms = (1/Y.shape[1] * np.sum((Y-AL)**2))**1/2
    return rms


part1 = True
if part1:
    # First part of the assignment

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'energy_efficiency_data.csv'))  
    glazing_one_hot = pd.get_dummies(data['Glazing Area Distribution']).rename(columns={0:"glazing_dist_0",1:"glazing_dist_1",2:"glazing_dist_2",3:"glazing_dist_3",4:"glazing_dist_4",5:"glazing_dist_5"})
    data = data.drop('Glazing Area Distribution',axis = 1)
    data = data.join(glazing_one_hot)

    orientation_one_hot = pd.get_dummies(data['Orientation']).rename(columns={2:"orient_2",3:"orient_3",4:"orient_4",5:"orient_5"})

    data = data.drop('Orientation',axis = 1)
    data = data.join(orientation_one_hot)

    # Normalization of terms
    data['Surface Area'] /= max(data['Surface Area'])
    data['Wall Area'] /= max(data['Wall Area'])
    data['Roof Area'] /= max(data['Roof Area'])
    data['Overall Height'] /= max(data['Overall Height'])

    train, test = train_test_split(data, test_size=0.25)
    X_train = train.drop('Heating Load', axis = 1).to_numpy(np.float64).T



    Y_train = train['Heating Load'].to_numpy(np.float64).T
    Y_train = Y_train.reshape(1,Y_train.shape[0])

    X_test = test.drop('Heating Load', axis = 1).to_numpy(np.float64).T
    Y_test = test['Heating Load'].to_numpy(np.float64).T
    Y_test = Y_test.reshape(1,Y_test.shape[0])
    layers_dims = [X_train.shape[0],20,20,1]
    last_activation = "none"
    parameters = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims,  learning_rate = 0.1, num_iterations = 10000)


else:
    # Second part of the assignment

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'ionosphere_data.csv'))  
    train, test = train_test_split(data, test_size=0.20)

    X_train = train.iloc[:,:-1].to_numpy(np.float64).T
    Y_train = train.iloc[:,-1:].to_numpy()
    Y_train = (Y_train == "g").astype(np.float64)
    Y_train = Y_train.reshape(1,Y_train.shape[0])

    X_test = test.iloc[:,:-1].to_numpy(np.float64).T
    Y_test = test.iloc[:,-1:].to_numpy()
    Y_test = (Y_test == "g").astype(np.float64)
    Y_test = Y_test.reshape(1,Y_test.shape[0])

    print("X_train.shape", X_train.shape)
    print(Y_train)
    layers_dims = [X_train.shape[0],20,20,1]
    last_activation = "sigmoid"
    parameters = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims, last_activation, learning_rate = 0.1, num_iterations = 10000)


predict(X_test, Y_test, parameters)

