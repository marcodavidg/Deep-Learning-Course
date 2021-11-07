import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import warnings

np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.8
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


def L_model_forward(X, parameters, last_activation, plot_latent):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    last_cache = None
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        last_cache = cache
        caches.append(cache)

    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation=last_activation)
    if plot_latent is not None and False:
        Z = last_cache[1]
        if Z.shape[0] == 2:
            print(Z.shape)
            plt.scatter(Z[0,:], Z[1,:], Z[2,:], c = (AL).astype(np.float64), cmap='cool')
            plt.title("2D Feature")
            plt.show()
        else:
            # Creating dataset
            z = Z[0,:]
            x = Z[1,:]
            y = Z[2,:]
             
            # Creating figure
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
             
            # Creating plot
            ax.scatter3D(x, y, z, c = (AL).astype(np.float64), cmap='cool')
            plt.title("3D Feature")
             
            # show plot
            plt.show()

    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, cost_formula = 'MSE'):
    m = Y.shape[1]
    if cost_formula == 'MSE':    
        # Part 1
        cost = np.sum((Y-AL)**2)
    else:
        # Part 2
        """For a Bernoulli distribution, the cross-entropy error function can be written like this, because K = 2."""
        cost = - np.sum(np.dot(np.log(AL),Y.T) + np.dot(np.log(1-AL),(1-Y).T))
    cost = np.squeeze(cost)
    
    return cost

def linear_backward(dZ, A_prev, W, b):
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

def linear_activation_backward(dA, A_prev, W, b, Z, batch_size, activation):
    
    if activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    else:
        dZ = dA

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, parameters, last_activation, learning_rate, batch_size = 200):
    
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    m = AL.shape[1]
    p = np.random.permutation(AL.shape[1])

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        grads = {}
        batch_indexes = p[start:end - 1]
        dAL = np.zeros(Y.shape)
        if last_activation == "none":
            # Part 1
            dAL[:, batch_indexes] = -(Y[:, batch_indexes] - AL[:, batch_indexes]) 
            # dAL[:, batch_indexes] = (AL[:, batch_indexes] - Y[:, batch_indexes]) # derivative of the loss function
        else:
            # Part 2
            dAL[:, batch_indexes] = -(np.divide(Y[:, batch_indexes], AL[:, batch_indexes]) - np.divide(1 - Y[:, batch_indexes], 1 - AL[:, batch_indexes])) #

        current_cache = caches[-1]

        linear_cache, activation_cache = current_cache
        A_prev, W, b = linear_cache
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, A_prev, W, b, activation_cache, batch_size, activation = last_activation)
        

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            linear_cache, activation_cache = current_cache
            A_prev, W, b = linear_cache

            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], A_prev, W, b, activation_cache, batch_size, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        L = len(parameters) // 2 # number of layers in the neural network
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters



def L_layer_model(X_test, Y_test, X, Y, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations, print_cost = True):
  
    costs = []  
    costs_test = [] # keep track of cost
    rms_costs = [] 
    parameters = initialize_parameters_deep(layers_dims)
    lastAL = None
    last_cost_test = None
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters, last_activation, i if (i % 5000 == 0) else None)
        learning_rate = max(min_lr, learning_rate - decay)
        lastAL = AL
        
        parameters = L_model_backward(AL, Y, caches, parameters, last_activation, learning_rate)
        if print_cost and i % 1 == 0:
            cost = compute_cost(AL, Y, cost_formula)
            print ("LR: %f Cost after iteration %i: %f" %(learning_rate, i, cost))
            costs.append(cost)
        if print_cost and i % 100 == 0:
            AL, caches = L_model_forward(X_test, parameters, last_activation, None)
            cost_test = compute_cost(AL, Y_test, cost_formula)
            costs_test.append(cost_test)
            if last_cost_test is not None and (last_cost_test - cost_test) < -0.1:
                print("Stopped at ", i," because ", last_cost_test, "vs", cost_test)
                break
            last_cost_test = cost_test
        
    fig = plt.figure(figsize=(5,5))
    rows = 1
    columns = 2
    print(np.round(lastAL,2))
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

def predict(X,Y,pretrained_weights,last_activation,cost_formula):
    AL, caches = L_model_forward(X, parameters,last_activation,False)
    cost = compute_cost(AL, Y, cost_formula)
    rms = computeRMS(AL, Y)
    print("Prediccion = ", AL)
    print(Y)

    # x = ((AL>0.5).astype(np.float64))**2 - Y
    # print("error",x.sum(),100 - x.sum()/Y.shape[1])
    print("Cost",cost)
    print("RMS", rms)

def computeRMS(AL,Y):
    rms = (1/Y.shape[1] * np.sum((Y-AL)**2))**1/2
    return rms


last_activation = ""
cost_formula = ""
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

    # 0.62,808.50,367.50,220.50,3.50,4.00,0.40,5.00,16.48,16.61

    X_train = np.array([0.05,0.10,0.1,0.20,0.21,0.12,5.12,0.25]).reshape(2,4)
    Y_train = np.array([100.07,0.84,1,-0.16]).reshape(1,4)
    X_test = X_train
    Y_test = Y_train

    layers_dims = [X_train.shape[0],5,5,1]
    last_activation = "none"
    cost_formula = "MSE"
    min_lr = 0.0001
    decay = 0.00000
    learning_rate = 0.0001
    num_iterations = 60000
    # parameters = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations)


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

    # X_train = np.array([0.05,0.10,0.1,0.20,0.21,0.12,5.12,0.25,1.12,3.25,33.12,-5.25]).reshape(2,6)
    # Y_train = np.array([0,0,1,1,0,0]).reshape(1,6)
    # Y_test = Y_train
    # X_test = X_train
    # plt.scatter(X[1,:],X[0,:], color="red")
    # plt.show()


    # layers_dims = [X_train.shape[0], 7, 3, 1]
    layers_dims = [X_train.shape[0], 3, 2, 1]
    last_activation = "sigmoid"
    cost_formula = "Cross Entropy"
    min_lr = 0.01
    decay = 0.00001
    learning_rate = 0.1
    num_iterations = 15000

parameters = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations)
print("layers_dims", layers_dims)
print("min_lr", min_lr)
print("decay", decay)
print("learning_rate",learning_rate)
print("iterations", num_iterations)
print("part1",part1)
predict(X_test[:,1:9], Y_test[:,1:9], parameters, last_activation, cost_formula)

