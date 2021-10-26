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

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1] + 1) * 0.01
        # parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))


        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1] + 1))
        # assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        # print("W" + str(l) + ":" + str(parameters['W' + str(l)].shape))

        
    return parameters


def linear_forward(A, W):

    # print("np.dot = W ", W.shape, " A ", A.shape)
    Z = np.dot(W, A)
    # print("resulting Z ", Z.shape)
    # print("ZZ" + str(Z.shape))

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W)
    
    return Z, cache


def relu(Z):
    return np.maximum(0, Z), Z
    # return 1/(1+np.exp(-Z))

def relu_backward(dA, Z):
    return np.multiply(np.multiply(Z, (1 - Z)), dA)

    # # print("--dA", dA)
    # # print("--Z", Z)
    # # dA = dA[]
    # dZ = np.array(dA, copy = False)
    # temp = np.r_[np.ones((1, Z.shape[1]), dtype=bool), Z<=0]
    # dZ[temp] = 0
    # # print("--temp", dZ.shape)
    # return dZ[1:,:]

def none_backward(dA, Z):
    return np.array(dA, copy = True)

def linear_activation_forward(A_prev, W, activation="relu"):
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
        Z, linear_cache = linear_forward(A_prev, W)
        A, activation_cache = relu(Z)

    elif activation == "none":
        Z, linear_cache = linear_forward(A_prev, W)
        A = Z
        activation_cache = Z 
        
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
    L = len(parameters)                  # number of layers in the neural network


    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        w = np.ones((A.shape[0]+1, A.shape[1]))
        w[1:,:] = A
        A = w
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)],  
                                             activation='relu')
        caches.append(cache)
        
    w = np.ones((A.shape[0]+1, A.shape[1]))
    w[1:,:] = A
    A = w
    A_prev = A 
    
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)],
                                          activation='none')
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):
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
    A_prev, W = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1/m * np.dot(dZ, cache[0].T)
    # db = 1/m * np.squeeze(np.sum(dZ, axis=0, keepdims=True))
    # print("A_prev",A_prev.shape)
    # print("cache[1].T",cache[1].T.shape)
    # print("dZ",dZ.shape)
    dA_prev = np.dot(cache[1].T, dZ)
    # print("dA_prev",dA_prev.shape)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    
    return dA_prev, dW


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
    


    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = none_backward(dA, activation_cache)

    dA_prev, dW = linear_backward(dZ, linear_cache)


    return dA_prev, dW


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
    # Initializing the backpropagation
    dAL = (AL - Y)
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "None")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]
        dA_prev_temp, dW_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        # grads["db" + str(l + 1)] = db_temp

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
    
    L = len(parameters) #// 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        # parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        ### END CODE HERE ###
    
    return parameters



def model_train(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False):
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
    
    # Parameters initialization. (≈ 1 line of code)
    
    parameters = initialize_parameters_deep(layers_dims)

    rms = 0
    lastAL = None
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            rms = computeRMS(AL, Y)
            print ("Cost after iteration %i: %f" %(i, rms))
            costs.append(rms)
            lastAL = AL
            clearConsole = lambda: print('\n' * 150)
            clearConsole()

            print(grads)
            print(i)
            
    print(lastAL)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
       
    return parameters

def model_test(X,Y,parameters):           
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

# print("Xtrain",X_train.shape)
# print("Xtest",X_test.shape)
# print("Ytrain",Y_train.shape)
# print("Ytest",Y_test.shape)

X = np.array([0.05,0.10,0.1,0.20]).reshape(2,2)
Y = np.array([0.07,0.84]).reshape(1,2)



# print("X", X.shape)
# print("Y", Y.shape)
layers_dims = [X.shape[0],10,10,1]
parameters = model_train(X, Y, layers_dims, learning_rate=0.1, num_iterations = 250000, print_cost = True)
# print(parameters['W1'].shape)
# model_test(X_test, Y_test, parameters)
print(parameters)

