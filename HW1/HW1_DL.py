import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    # for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        print("W" + str(l) + ":" + str(parameters['W' + str(l)].shape))


        
    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def relu(Z):
    # return np.maximum(0, Z), Z
    return 1 / (1 + np.exp(-Z)),Z

def linear_activation_forward(A_prev, W, b, activation="relu"):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "none":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        # A = Z
        # activation_cache = Z 
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

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
                                          activation='none')
    caches.append(cache)
        
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y, formula = 'MSE'):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    # Compute loss from aL and y.
    if formula == 'MSE':    
        cost = np.sum((Y-AL)**2)
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1/m * np.dot(dZ, cache[0].T)
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    # x=1/0
    # print("----------------------------------------------------")
    # print(dZ)
    # print(db)
    # print("----------------------------------------------------")
    dA_prev = np.dot(cache[1].T, dZ)
    ### END CODE HERE ###
    # print("db",dZ)
    # x = 1/0
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    # assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def relu_backward(dA, x):
    gv = 1 / (1 + np.exp(-x))  
    return np.multiply(np.multiply(gv, (1 - gv)), dA)

    # dZ = np.array(dA, copy = True)
    # dZ[Z <= 0] = 0
    # # print("dA",dA.shape,"dZ",dZ.shape)

    # return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    # print("linear")
    # print(linear_cache)
    # print("activation_cache")
    # print(activation_cache)

    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        ### END CODE HERE ###
    else:
        dZ = np.copy(dA)
    # print("dZ")
    # print(dZ)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    # print("dA_prev")
    # print(dA_prev)
    # print("dW")
    # print(dW)
    # print("db")
    # print(db)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # print("Y",Y.shape)
    # print("L")
    # print(L)
    # print("AL")
    # print(AL)
    # Initializing the backpropagation
    dAL = AL-Y 
    # print("dAL")
    # print(dAL)
    
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "None")
    # print(grads)
    # print(L-1)

    for l in reversed(range(L-1)):
        # print("L---" + str(l))
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    # print(grads)    
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    # print("grads")
    # print(grads)

    # print("parameters before")
    # print(parameters)
    for l in range(L):
        # print("***********operacion shapes")
        # print(parameters["b" + str(l + 1)].shape)
        # print(grads["db" + str(l + 1)].shape)
        # print("****")
        # print(grads["dW" + str(l + 1)])
        # print(parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)])
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        # print("b y db", parameters["b" + str(l + 1)].shape, grads["db" + str(l + 1)].shape)
        
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]#[:,0]
        # x=1/0
    ### END CODE HERE ###
    
    # print("parameters after")
    # print(parameters)
    return parameters



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    rms_costs = [] 
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    print(parameters["b1"])
    ### END CODE HERE ###
    # Loop (gradient descent)
    lastAL = None
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            rms = computeRMS(AL, Y)
            rms_costs.append(rms)
            costs.append(cost)
            lastAL = AL
            clearConsole = lambda: print('\n' * 150)
            clearConsole()

            print(grads)
            print(i)

    print(AL)            
    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    # # plot the cost
    # plt.plot(np.squeeze(rms_costs))
    # plt.ylabel('rms_costs')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters

def predict(X,Y,pretrained_weights):
    AL, caches = L_model_forward(X, parameters)
    cost = compute_cost(AL, Y)
    rms = computeRMS(AL, Y)
    print ("Cost after prediction test data %f" %(rms))

def computeRMS(AL,Y):
    rms = (1/Y.shape[1] * np.sum((Y-AL)**2))**1/2
    return rms

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'energy_efficiency_data.csv'))  
glazing_one_hot = pd.get_dummies(data['Glazing Area Distribution']).rename(columns={0:"glazing_dist_0",1:"glazing_dist_1",2:"glazing_dist_2",3:"glazing_dist_3",4:"glazing_dist_4",5:"glazing_dist_5"})
data = data.drop('Glazing Area Distribution',axis = 1)
data = data.join(glazing_one_hot)

orientation_one_hot = pd.get_dummies(data['Orientation']).rename(columns={2:"orient_2",3:"orient_3",4:"orient_4",5:"orient_5"})

data = data.drop('Orientation',axis = 1)
data = data.join(orientation_one_hot)

# data['Surface Area'] /= max(data['Surface Area'])
# data['Wall Area'] /= max(data['Wall Area'])
# data['Roof Area'] /= max(data['Roof Area'])
# data['Overall Height'] /= max(data['Overall Height'])

train, test = train_test_split(data, test_size=0.25)
X_train = train.drop('Heating Load', axis = 1).to_numpy(np.float64).T



Y_train = train['Heating Load'].to_numpy(np.float64).T
Y_train = Y_train.reshape(1,Y_train.shape[0])

X_test = test.drop('Heating Load', axis = 1).to_numpy(np.float64).T
Y_test = test['Heating Load'].to_numpy(np.float64).T
Y_test = Y_test.reshape(1,Y_test.shape[0])

# print("Xtrain",X_train.shape)
# print("Xtest",X_test.shape)
# print("Ytrain",Y_train.shape)
# print("Ytest",Y_test.shape)


# print("X", X.shape)
# print("Y", Y.shape)
# layers_dims = [X_train.shape[0],10,1]
# parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.01, num_iterations = 250000, print_cost = True)
# print(parameters['W1'].shape)
# model_test(X_test, Y_test, parameters)




X = np.array([0.05,0.10,0.1,0.20,0.21,0.12,5.12,0.25]).reshape(2,4)
Y = np.array([0.07,0.84,1,0.16]).reshape(1,4)
layers_dims = [X_train.shape[0],20,20,1]
parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate = 0.001, num_iterations = 200, print_cost = True)
print(parameters["b1"])
# predict(X_test, Y_test, parameters)

