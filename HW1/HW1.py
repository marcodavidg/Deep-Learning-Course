import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import warnings
from matplotlib.lines import Line2D

np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    """Initialize the weights and biases for the model. The weights are set in a very small value and biases are set to 0."""
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.8
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


def linear_forward(A, W, b):
    """Compute the linear operation Z = X*W + b. The values are stored in a cache for the backprop procedure."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    # Apply the sigmoid to a vector
    return 1 / (1 + np.exp(-Z)), Z 

def relu(Z):
    # Apply the ReLU function to a vector
    return np.maximum(0, Z), Z

def linear_activation_forward(A_prev, W, b, activation="relu"):
    """From the previous activation function output and this layer's weights and biases, return this layer's activation output
    A = g(Z)
    g = sigmoid / ReLU / none
    Z = A_prev * W + b
    """

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
    """Do the forward pass of the neural network, 
    while saving the outputs of each stage to a cache fo save computation time during the backprop."""
    caches = []
    A = X # A0 will be the input layer
    L = len(parameters) // 2 # number of layers in the neural network
    
    last_cache = None

    # Iterate through all the hidden layers
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        last_cache = cache
        caches.append(cache)

    # Compute the output layer. AL will be the final prediction of the NN.
    AL, cache = linear_activation_forward(A, 
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation=last_activation)

    # Compute the second to last layer for the classification problem
    if plot_latent is not None:
        Z = last_cache[1]
        legend_elements = [Line2D([0], [0], marker='o', color='cyan', label='b',
                      markerfacecolor='cyan', markersize=15),
                            Line2D([0], [0], marker='o', color='magenta', label='g',
                      markerfacecolor='red', markersize=15)]

        # If the layer is composed of 2 neurons, plot a 2D graph, otherwise, plot a 3D graph.
        if Z.shape[0] == 2:
            print(AL>0.5)
            plt.scatter(Z[0,:], Z[1,:], c = (AL>0.5).astype(np.float64), cmap='cool')
            plt.title("2D Feature Epoch #" + str(plot_latent))
            plt.legend(handles=legend_elements, loc='upper right')
            plt.show()
        else:
            # Creating dataset
            z = Z[0,:]
            x = Z[1,:]
            y = Z[2,:]
            
            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            
            ax.scatter3D(x, y, z, c = (AL>0.5).astype(np.float64), cmap='cool')
            plt.title("3D Feature Epoch #" + str(plot_latent))
            plt.legend(handles=legend_elements, loc='best')
            plt.show()

    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, cost_formula = 'MSE'):
    """Compute the cost of the predictions"""
    m = Y.shape[1]
    if cost_formula == 'MSE':    
        # Regression problem
        cost = np.sum((Y-AL)**2)
    else:
        # Classification problem
        """For a Bernoulli distribution, the cross-entropy error function can be written like this, because K = 2."""
        cost = - np.sum(np.dot(np.log(AL),Y.T) + np.dot(np.log(1-AL),(1-Y).T))
    cost = np.squeeze(cost)
    
    return cost

def linear_backward(dZ, A_prev, W, b):
    # The derivatives of the linear part of the operations

    m = A_prev.shape[1]
    dW = np.matmul(dZ,A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.matmul(W.T, dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, Z):
    # The derivative of the ReLU 

    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    # The derivative of the Sigmoid
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def linear_activation_backward(dA, A_prev, W, b, Z, batch_size, activation):
    # The derivative of the activation function

    if activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    else:
        dZ = dA

    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, parameters, last_activation, learning_rate, batch_size):
    """Backpropagation algorithm. This method does not only compute the gradients, 
    but also executes the stochastic gradient descent with these results.
    """

    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    m = AL.shape[1] # Total examples
    p = np.random.permutation(AL.shape[1]) # Vector to randomize the examples.

    for start in range(0, m, batch_size):
        # Stochastic gradient descent

        end = min(start + batch_size, m)
        grads = {}

        # batch_indexes has the random indexes of the elements that will be taken into consideration in this step of the stochastic gradient descent 
        batch_indexes = p[start:end - 1]

        dAL = np.zeros(Y.shape)

        # dAL = Derivative of the loss function
        if last_activation == "none":
            # Regression
            dAL[:, batch_indexes] = -(Y[:, batch_indexes] - AL[:, batch_indexes]) 
        else:
            # Classification
            dAL[:, batch_indexes] = -(np.divide(Y[:, batch_indexes], AL[:, batch_indexes]) - np.divide(1 - Y[:, batch_indexes], 1 - AL[:, batch_indexes]))


        # First step of backprop
        current_cache = caches[-1]
        linear_cache, activation_cache = current_cache
        A_prev, W, b = linear_cache
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, A_prev, W, b, activation_cache, batch_size, activation = last_activation)
        
        # Compute the gradients for the rest of layers
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            linear_cache, activation_cache = current_cache
            A_prev, W, b = linear_cache

            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], A_prev, W, b, activation_cache, batch_size, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        # Perform a step of gradient descent with the gradients that were computed earlier
        number_layers = len(parameters) // 2
        for l in range(number_layers):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def L_layer_model(X_test, Y_test, X, Y, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations, latent = False, batch_size = 200, print_cost = True):
    """Train a neural network and plot the training and test loss during training."""  
    costs = []  
    costs_test = []
    rms_costs = [] 
    lastAL = None
    last_cost_test = None

    # Initialize the weights and biases
    parameters = initialize_parameters_deep(layers_dims)

    # Training process
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters, last_activation, i if (i % 5000 == 0 and latent) else None) # Every 5000 steps, plot the second to last layer
        lastAL = AL

        # Learning rate with decay
        learning_rate = max(min_lr, learning_rate - decay)
        
        # Get updated parameters after gradient descent
        parameters = L_model_backward(AL, Y, caches, parameters, last_activation, learning_rate, batch_size)
        
        # Compute the cost
        cost = compute_cost(AL, Y, cost_formula)

        # Print the cost of the epoch
        print ("LR: %f Cost after iteration %i: %f" %(learning_rate, i, cost))
        costs.append(cost)

        # Every 100 epochs, compute the loss for the test data
        if print_cost and i % 100 == 0:
            AL, caches = L_model_forward(X_test, parameters, last_activation, None)
            cost_test = compute_cost(AL, Y_test, cost_formula)
            costs_test.append(cost_test)
            if last_cost_test is not None and (last_cost_test - cost_test) < -0.1:
                # Early stopping
                print("Stopped at ", i," because ", last_cost_test, "vs", cost_test)
                break
            last_cost_test = cost_test

    # Plot the training and test loss during the whole process
    fig = plt.figure(figsize=(15,15))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.plot(np.squeeze(costs_test))
    plt.yscale('log')
    plt.ylabel('Cost - Test data')
    plt.xlabel('Epoch (per hundreds)')
    plt.title("Test loss during training")

    fig.add_subplot(rows, columns, 2)
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost - Training data')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.title("Training loss during training")

    plt.show()
    return parameters, lastAL

def predict(X, Y, pretrained_weights, last_activation, cost_formula):
    # Get the results for a single forward pass with some pre-trained weights
    AL, caches = L_model_forward(X, parameters, last_activation, False)
    return AL

def computeRMS(AL,Y):
    # Compute the RMS Error
    rms = (1/Y.shape[1] * np.sum((Y-AL)**2))**1/2
    return rms

def plot_predictions(plot_name, AL, Y):
    # Plot the predictions of the classification problem

    fig = plt.figure(figsize=(10,10))
    rows = 1
    columns = 1
    fig.add_subplot(rows, columns, 1)
    plt.plot(np.arange(AL.shape[1]), AL[0,:], color="blue", label='Predictions')
    plt.plot(np.arange(Y.shape[1]), Y[0,:], color="red", label='Ground-truth')
    plt.ylabel('Prediction')
    plt.xlabel('#th case')
    plt.legend()
    plt.title(plot_name)
    plt.show()


def classification_performance(AL, Y):
    # Meassure the correct rate of the classification problem

    missed_positives = ((((AL>0.5).astype(np.float64) - Y) == -1).astype(np.float64)).sum() # Tendria que haber sido 1 pero fue 0
    false_positives = ((((AL>0.5).astype(np.float64) - Y) == 1).astype(np.float64)).sum() # Tendria que haber sido 0 pero fue 1
    correct_predictions = ((((AL>0.5).astype(np.float64) - Y) == 0).astype(np.float64)).sum() # Resultado correcto
    print("Correct predictions",correct_predictions)
    print("False positives",false_positives)
    print("Missed positives",missed_positives)
    print(Y.shape[1])
    print("Accuracy rate",float(correct_predictions)/float(Y.shape[1]))


last_activation = ""
cost_formula = ""
part1 = True
if part1:
    # First part of the assignment. A regression problem.

    # Load of data and one-hot encoding
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

    # Split data 85%-25%
    train, test = train_test_split(data, test_size=0.25)

    X_train = train.drop('Heating Load', axis = 1).to_numpy(np.float64).T
    Y_train = train['Heating Load'].to_numpy(np.float64).T
    Y_train = Y_train.reshape(1,Y_train.shape[0])

    X_test = test.drop('Heating Load', axis = 1).to_numpy(np.float64).T
    Y_test = test['Heating Load'].to_numpy(np.float64).T
    Y_test = Y_test.reshape(1,Y_test.shape[0])

    # Hyperparameters 
    last_activation = "none"
    cost_formula = "MSE"
    min_lr = 0
    decay = 0
    learning_rate = 0.0001
    num_iterations = 10000
    batch_size = 200

    # Feature selection
    features_to_drop = [4, 5, 16, 3, 2, 0, 15]
    X_train = np.delete(X_train,features_to_drop,0)
    X_test = np.delete(X_test,features_to_drop,0)

    # Training
    layers_dims = [X_train.shape[0],5,5,1]
    parameters, AL = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations, False, batch_size)
    
    # Training and test RMS Error
    print("Features removed:", features_to_drop)
    print("Training RMS:", computeRMS(AL, Y_train))
    plot_predictions("Prediction for training data", AL, Y_train)

    predictions = predict(X_test, Y_test, parameters, last_activation, cost_formula)
    print("Test RMS:", computeRMS(predictions, Y_test))
    plot_predictions("Prediction for test data", predictions, Y_test)
else:
    # Second part of the assignment. A classification problem.

    # Load the data and encode the classes. g = 1, b = 0.
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'ionosphere_data.csv'))  
    train, test = train_test_split(data, test_size=0.20) #Split the data at 80%-20%

    X_train = train.iloc[:,:-1].to_numpy(np.float64).T
    Y_train = train.iloc[:,-1:].to_numpy()
    Y_train = (Y_train == "g").astype(np.float64)
    Y_train = Y_train.reshape(1,Y_train.shape[0])

    X_test = test.iloc[:,:-1].to_numpy(np.float64).T
    Y_test = test.iloc[:,-1:].to_numpy()
    Y_test = (Y_test == "g").astype(np.float64)
    Y_test = Y_test.reshape(1,Y_test.shape[0])


    # Hyperparameters of the model 
    layers_dims = [X_train.shape[0], 3, 3, 1]
    last_activation = "sigmoid"
    cost_formula = "Cross Entropy"
    min_lr = 0
    decay = 0
    learning_rate = 0.01
    num_iterations = 20001 # 

    # Training
    parameters, AL = L_layer_model(X_test, Y_test, X_train, Y_train, layers_dims, last_activation, cost_formula, min_lr, decay, learning_rate, num_iterations)
    
    # Correct rate for training data
    classification_performance(AL, Y_train)
    predictions = predict(X_test, Y_test, parameters, last_activation, cost_formula)
    
    # Correct rate for test data
    print("Now for test data")
    classification_performance(predictions, Y_test)

 
# Printing the settings used for the execution
print("Dimensions of NN", layers_dims)
print("Min_lr", min_lr)
print("Decay", decay)
print("Learning rate",learning_rate)
print("#Iterations", num_iterations)
print("Was this part 1 of the assignment? ",part1)


