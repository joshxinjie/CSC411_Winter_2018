from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")
RAND_SEED = 7
P5_WEIGHTS_CSV_NAME = "part5_weights.csv"
P5_BIAS_CSV_NAME = "part5_bias.csv"
'''
#Display the 150-th "5" digit from the training set
imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
show()
'''

''' ##### Part 1 ##### '''
def part1():
    print "Running Part 1 ..."
    f, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            fig = ax[i,j].imshow(M["train"+str(i)][j].reshape((28,28)), cmap=cm.gray)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    plt.show()
    f.savefig("part1")
    print "End of Part 1"
''' ##### End of Part 1 ##### '''

''' ##### Part 2 ##### '''
def linear_neuron(X, weights, bias):
    '''
    X: is the input matrix of images - 784xM matrix
    weights: is the matrix of weights - Nx784 matrix
    bias: an Nx1 matrix
    
    Returns an NxM matrix
    
    where N is number of outputs (10 for digits) and M is number of images
    '''
    return np.dot(weights, X) + bias

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case (10 for digits), and M
    is the number of cases (number of images)'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
''' ##### End of Part 2 ##### '''

''' ##### Part 3 ##### '''
def gradient(X, weights, bias, Y_):
    ''' 
    X: the input matrix of images - 784xM matrix
    weights: the matrix of weights - Nx784 matrix
    bias: an Nx1 matrix 
    Y_: the one-hot encoding matrix -NxM matrix
    
    Returns
    grad_w: an Nx784 matrix (same dimensions as weights) for gradient w.r.t. weights
    grad_b: an Nx1 matrix (same dimensions as weights) for gradient w.r.t. bias
    
    where N is number of outputs (10 for digits) and M is number of images
    '''
    L = linear_neuron(X, weights, bias)
    Y = softmax(L)
    grad_w = np.dot((Y-Y_), X.T)
    
    grad_b = np.dot((Y-Y_), np.ones((Y.shape[1], 1)))  #Check!!!
    
    return grad_w, grad_b
    
def finite_difference(X, weights, bias, Y_, h):
    '''
    X: the input matrix of images: 784xM matrix
    weights: the matrix of weights: Nx784 matrix
    bias: an Nx1 matrix
    Y_: the one-hot encoding matrix: NxM matrix
    
    returns an Nx784 matrix (same dimensions as weights) for gradient w.r.t. weights
    and an Nx1 matrix (same dimensions as weights) for gradient w.r.t. bias
    
    where N is number of outputs (10 for digits) and M is number of images
    '''
    finite_grad_w = np.zeros(weights.shape)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            H = np.zeros(weights.shape)
            H[i][j] = h
            Y_plusH = softmax(linear_neuron(X, weights+H, bias))
            Y_minusH = softmax(linear_neuron(X, weights-H, bias))
            finite_grad_w[i][j] = (NLL(Y_plusH, Y_) - NLL(Y_minusH, Y_))/(2*h)
    finite_grad_bias = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        H = np.zeros(weights.shape[0])
        H[i] = h
        H = H.reshape(bias.shape[0], bias.shape[1])
        Y_plusH = softmax(linear_neuron(X, weights, bias+H))
        Y_minusH = softmax(linear_neuron(X, weights, bias-H))
        finite_grad_bias[i] = (NLL(Y_plusH, Y_) - NLL(Y_minusH, Y_))/(2*h)
    return finite_grad_w, finite_grad_bias.reshape(3,1)

def part3():
    print "Running Part 3 ..."
    random.seed(0)
    h = 0.00001
    N = 3
    pixels = 5
    M = 5
    weights = rand(N, pixels)
    bias = rand(N,1)
    X = rand(pixels, M)
    Y_ = array([[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0]])
    act_grad_w, act_grad_b = gradient(X, weights, bias, Y_)
    est_grad_w, est_grad_b = finite_difference(X, weights, bias, Y_, h)
    print "The actual gradient with respect to weights is"
    print act_grad_w
    print "Using finite difference, the estimated gradient with respect to weights is"
    print est_grad_w
    print "The actual gradient with respect to bias is"
    print act_grad_b
    print "Using finite difference, the estimated gradient with respect to bias is"
    print est_grad_b
    print "End of Part 3"
''' ##### End of Part 3 ##### '''

''' ##### Part 4 ##### '''
def setup():
    '''
    Splits training images into a training set containing 80% of the images and
    a validation set containing 20% of the images, and creates the test set of
    images. The images will be normalized by dividing every pixel by 255.0.
    Also creates the one-hot-encoding for the test, validation and training set.
    Returns
    X: training set of images: 784x(0.8*M) matrix
    X_valid: validation set of images: 784x(0.2*M) matrix
    X_test: test set of images: 784x10000 matrix
    Y_train: one-hot encoding for training set: 10x(0.8*M) matrix
    Y_valid: one-hot encoding for validation set: 10x(0.2*M) matrix
    Y_test: one-hot encoding for test set: 10x10000 matrix
    '''
    #Create training and validation sets
    np.random.seed(RAND_SEED)
    for digit in range(10):
        num_images = len(M["train"+str(digit)])
        images = M["train"+str(digit)]/255.0 # Normalize the images
        np.random.shuffle(images)
        train_images = images[0:int(num_images*0.8)]
        valid_images = images[int(num_images*0.8):]
        train_one_hot = np.zeros((int(num_images*0.8),10))
        valid_one_hot = np.zeros((num_images-int(num_images*0.8),10))
        train_one_hot[:,digit] = 1
        valid_one_hot[:,digit] = 1
        if digit == 0:
            X = np.vstack([train_images])
            X_valid = np.vstack([valid_images])
            Y_train = np.vstack([train_one_hot])
            Y_valid = np.vstack([valid_one_hot])
        else:
            X = np.vstack((X, train_images))
            X_valid = np.vstack((X_valid, valid_images))
            Y_train = np.vstack((Y_train, train_one_hot))
            Y_valid = np.vstack((Y_valid, valid_one_hot))
    
    # Create test set
    X_test = (np.vstack((M['test0'], M['test1'], M['test2'], M['test3'],\
                 M['test4'], M['test5'], M['test6'], M['test7'],\
                 M['test8'], M['test9'])).T)/255.0
    for digit in range(10):
        num_images = len(M["test"+str(digit)])
        test_one_hot = np.zeros((num_images,10))
        test_one_hot[:,digit] = 1
        if digit == 0:
            Y_test = np.vstack([test_one_hot])
        else:
            Y_test = np.vstack((Y_test, test_one_hot))
    X = X.T
    X_valid = X_valid.T
    Y_train = Y_train.T
    Y_valid = Y_valid.T
    Y_test = Y_test.T
    return X, X_valid, X_test, Y_train, Y_valid, Y_test

def gradient_descent(X, weights, bias, Y_, alpha, X_valid, Y_valid, max_iter, eps=1e-5):
    diff_weights = np.array([eps + 1.0])  # make sure the initial loop goes through
    i = 0
    train_perf_record = []
    valid_perf_record = []
    train_cost_record = []
    valid_cost_record = []
    iterations = []
    while np.linalg.norm(diff_weights) > eps and i < max_iter:
        prev_weights = weights.copy()
        grad_w, grad_b = gradient(X, weights, bias, Y_)
        weights -= alpha*grad_w
        bias -= alpha*grad_b
        if i % 100 == 0:
            train_perf_record.append(performance(X, weights, bias, Y_))
            valid_perf_record.append(performance(X_valid, weights, bias, Y_valid))
            train_cost_record.append(cost(X, weights, bias, Y_))
            valid_cost_record.append(cost(X_valid, weights, bias, Y_valid))
            iterations.append(i)
        i += 1
        diff_weights = weights - prev_weights
    return weights, bias, train_perf_record, valid_perf_record, train_cost_record, valid_cost_record, iterations
        
def performance(X, weights, bias, Y_):
    L = linear_neuron(X, weights, bias)
    Y = softmax(L)
    Y = Y.T # Each row now correponds to an image
    Y_ = Y_.T # Each row now correponds to an image
    accuracy = 0
    for i in range(len(Y)):
        # If index of maximium element in each row (image) or predictions is equal to 
        # the index of 1 in the corresponding row of the label matrix
        if np.argmax(Y[i]) == np.argmax(Y_[i]):
            accuracy += 1
    return accuracy/float(len(Y))

def cost(X, weights, bias, Y_):
    L = linear_neuron(X, weights, bias)
    Y = softmax(L)
    return NLL(Y,Y_)

def part4():
    print "Running Part 4 ..."
    X, X_valid, X_test, Y_, Y_valid, Y_test = setup()
    init_bias = 1
    max_iter = 5000
    init_weights = np.zeros((10,784))
    
    # Grid Search for Optimal Learning Rate
    # Uncomment to Run Grid Search Cause it Takes Too Long
    '''
    learning_rates = np.arange(0.00005, 0.0003, 0.00005)
    train_perf_learn = []
    valid_perf_learn = []
    train_cost_learn = []
    valid_cost_learn = []
    for alpha in learning_rates:
        weights, bias, train_perf, valid_perf, train_cost, valid_cost, iterations = gradient_descent(X, init_weights, init_bias, Y_, alpha, X_valid, Y_valid, max_iter)
        train_perf_learn.append(train_perf[-1])
        valid_perf_learn.append(valid_perf[-1])
        train_cost_learn.append(train_cost[-1])
        valid_cost_learn.append(valid_cost[-1])
    
    learning_rates = learning_rates*10000
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(learning_rates, train_perf_learn, label="Training Performance")
    ax1.plot(learning_rates, valid_perf_learn, 'r-', label="Validation Performance")
    plt.xlabel("Learning Rates (x 10^-5)")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Rate of Correct Classification for MNIST Digits")
    plt.legend(loc=0)
    plt.savefig("Part4_Learning_Rate_Performance")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(learning_rates, train_cost_learn, label="Training Cost")
    ax2.plot(learning_rates, valid_cost_learn, 'r-', label="Validation Cost")
    plt.xlabel("Learning Rates (x 10^-5)")
    plt.ylabel("Cost")
    plt.title("Cost for MNIST Digits")
    plt.legend(loc=0)
    plt.savefig("Part4_Learning_Rate_Cost")  
    '''
    
    # Use Optimal learning Rate of 0.000005
    opt_alpha = 0.000005
    weights, bias, train_perf, valid_perf, train_cost, valid_cost, iterations = gradient_descent(X, init_weights, init_bias, Y_, opt_alpha, X_valid, Y_valid, max_iter)
    
    # Save the weights
    for i in range(10):
        w = weights[i,:]
        w = w.reshape((28, 28))
        imsave("Part4_digit"+str(i)+".jpg", w)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(iterations, train_perf, label="Training Performance")
    ax3.plot(iterations, valid_perf, 'r-', label="Validation Performance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Classification Accuracy Using Learning Rate of 0.000005")
    plt.legend(loc=0)
    plt.savefig("Part4_Final_Performance")
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(iterations, train_cost, label="Training Cost")
    ax4.plot(iterations, valid_cost, 'r-', label="Validation Cost")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Classification Cost Using Learning Rate of 0.000005")
    plt.legend(loc=0)
    plt.savefig("Part4_Final_Cost")
    
    test_performance = performance(X_test, weights, bias, Y_test)
    test_cost = cost(X_test, weights, bias, Y_test)
    
    print "Final Training Set Performance: "+str(train_perf[-1])
    print "Final Training Set Cost: "+str(train_cost[-1])
    print "Final Validation Set Performance: "+str(valid_perf[-1])
    print "Final Validation Set Cost: "+str(valid_cost[-1])
    print "Final Test Set Performance: "+str(test_performance)
    print "Final Test Set Cost: "+str(test_cost)
    
    print "End of Part 4"
    
''' ##### End of Part 4 ##### '''

''' ##### Start of Part 5 ##### '''
def mom_gradient_descent(X, weights, bias, Y_, alpha, X_valid, Y_valid, max_iter, eps=1e-5, gamma=0.9):
    Vw = np.zeros(weights.shape)
    Vb = np.zeros((weights.shape[0], 1))

    diff_weights = np.array([eps + 1.0])  # make sure the initial loop goes through
    i = 0
    train_perf_record = []
    valid_perf_record = []
    train_cost_record = []
    valid_cost_record = []
    iterations = []
    while np.linalg.norm(diff_weights) > eps and i < max_iter:
        prev_weights = weights.copy()
        grad_w, grad_b = gradient(X, weights, bias, Y_)
        Vw = gamma*Vw + alpha*grad_w
        weights -= Vw
        Vb = gamma*Vb + alpha*grad_b
        bias -= Vb
        if i % 100 == 0:
            train_perf_record.append(performance(X, weights, bias, Y_))
            valid_perf_record.append(performance(X_valid, weights, bias, Y_valid))
            train_cost_record.append(cost(X, weights, bias, Y_))
            valid_cost_record.append(cost(X_valid, weights, bias, Y_valid))
            iterations.append(i)
        i += 1
        diff_weights = weights - prev_weights
    return weights, bias, train_perf_record, valid_perf_record, train_cost_record, valid_cost_record, iterations

def part5():
    print "Running Part 5 ..."
    X, X_valid, X_test, Y_, Y_valid, Y_test = setup()
    init_bias = 1
    max_iter = 5000
    init_weights = np.zeros((10,784))
    
    # Grid Search for Optimal Learning Rate
    # Uncomment to Run Grid Search Cause it Takes Too Long
    '''
    learning_rates = np.arange(0.00005, 0.0003, 0.00005)
    train_perf_learn = []
    valid_perf_learn = []
    train_cost_learn = []
    valid_cost_learn = []
    for alpha in learning_rates:
        weights, bias, train_perf, valid_perf, train_cost, valid_cost, iterations = gradient_descent(X, init_weights, init_bias, Y_, alpha, X_valid, Y_valid, max_iter)
        train_perf_learn.append(train_perf[-1])
        valid_perf_learn.append(valid_perf[-1])
        train_cost_learn.append(train_cost[-1])
        valid_cost_learn.append(valid_cost[-1])
    
    learning_rates = learning_rates*10000
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(learning_rates, train_perf_learn, label="Training Performance")
    ax1.plot(learning_rates, valid_perf_learn, 'r-', label="Validation Performance")
    plt.xlabel("Learning Rates (x 10^-5)")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Rate of Correct Classification for MNIST Digits")
    plt.legend(loc=0)
    plt.savefig("Part5_Learning_Rate_Performance")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(learning_rates, train_cost_learn, label="Training Cost")
    ax2.plot(learning_rates, valid_cost_learn, 'r-', label="Validation Cost")
    plt.xlabel("Learning Rates (x 10^-5)")
    plt.ylabel("Cost")
    plt.title("Cost for MNIST Digits")
    plt.legend(loc=0)
    plt.savefig("Part5_Learning_Rate_Cost")  
    '''
    # Use Optimal learning Rate of 0.000005
    opt_alpha = 0.00001
    weights, bias, train_perf, valid_perf, train_cost, valid_cost, iterations = mom_gradient_descent(X, init_weights, init_bias, Y_, opt_alpha, X_valid, Y_valid, max_iter)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(iterations, train_perf, label="Training Performance")
    ax3.plot(iterations, valid_perf, 'r-', label="Validation Performance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Classification Accuracy Using Momentum Gradient Descent")
    plt.legend(loc=0)
    plt.savefig("Part5_Final_Performance")
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(iterations, train_cost, label="Training Cost")
    ax4.plot(iterations, valid_cost, 'r-', label="Validation Cost")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Classification Cost Using Momentum Gradient Descent")
    plt.legend(loc=0)
    plt.savefig("Part5_Final_Cost")
    
    test_performance = performance(X_test, weights, bias, Y_test)
    test_cost = cost(X_test, weights, bias, Y_test)
    
    print "Final Training Set Performance: "+str(train_perf[-1])
    print "Final Training Set Cost: "+str(train_cost[-1])
    print "Final Validation Set Performance: "+str(valid_perf[-1])
    print "Final Validation Set Cost: "+str(valid_cost[-1])
    print "Final Test Set Performance: "+str(test_performance)
    print "Final Test Set Cost: "+str(test_cost)

    # output the weights and bias to csv
    #pd.DataFrame(weights).to_csv(P5_WEIGHTS_CSV_NAME, index=False)
    #pd.DataFrame(bias).to_csv(P5_BIAS_CSV_NAME, index=False)

    print "End of Part 5"    
''' ##### End of Part 5 ##### '''

''' ##### Start of Part 6 ##### '''
def cost_weights_vary(X, weights, bias, Y_, w1, w2, col1, col2):
    new_weights = weights.copy()
    new_weights[:, col1] = w1
    new_weights[:, col2] = w2
    L = linear_neuron(X, new_weights, bias)
    Y = softmax(L)
    return NLL(Y,Y_)

def gradient_descent_2weights(X, weights, bias, Y_, alpha, col1, col2):
    grad_w, grad_b = gradient(X, weights, bias, Y_)
    weights[:, col1] -= alpha*grad_w[:, col1]
    weights[:, col2] -= alpha*grad_w[:, col2]

def gradient_descent_momentum_2weights(X, weights, bias, Y_, alpha, col1, col2, Vw, gamma=0.9):
    grad_w, grad_b = gradient(X, weights, bias, Y_)
    Vw = gamma*Vw + alpha*grad_w
    weights[:, col1] -= Vw[:, col1]
    weights[:, col2] -= Vw[:, col2]
    return Vw

def part6():
    # part 6 (a)
    print "Start of Part 6 ..."
    X, X_valid, X_test, Y_, Y_valid, Y_test = setup()
    weights = np.array(pd.read_csv(P5_WEIGHTS_CSV_NAME))
    bias = np.array(pd.read_csv(P5_BIAS_CSV_NAME))
    # select the centre two pixels
    col1 = 13 * 28 + 13
    col2 = col1 + 1

    w1 = weights[:, col1].copy()
    w2 = weights[:, col2].copy()

    # contour plot for the selected neuron
    neuron = 1  # can be changed to show other neurons
    density = 10  # can be changed to adjust the number of points in the contour plot
    width = 10.0  # can be changed to adjust the band width of the x and y axis
    x_axis = np.linspace(w1[1]-width, w1[1]+width, density)
    y_axis = np.linspace(w2[1]-width, w2[1]+width, density)
    z_axis = np.zeros((density, density))
    for i in range(density):
        for j in range(density):
            w1[neuron] = x_axis[j]
            w2[neuron] = y_axis[i]
            z_axis[i, j] = cost_weights_vary(X, weights, bias, Y_, w1, w2, col1, col2)
            # print i, j
    plt.figure()
    plt.contour(x_axis, y_axis, z_axis)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title("Contour Plot for the Cost Function of Neuron " + str(neuron))
    plt.savefig("Part 6(a) Contour Plot")

    # part 6 (b) (c)
    alpha_b = 0.0026  # learning rate for vanilla
    alpha_c = 0.0034  # learning rate for momentum
    k = 15  # k iterations for gradient descent

    weights_vanilla = weights.copy()
    w1_van_lst = []
    w2_van_lst = []
    # adjust w1 w2 away from minimum
    weights_vanilla[:, col1] -= width * 0.75
    weights_vanilla[:, col2] -= width * 0.75
    w1_van_lst.append(weights_vanilla.copy()[:, col1])
    w2_van_lst.append(weights_vanilla.copy()[:, col2])
    # take k steps of gradient descent
    for i in range(k):
        gradient_descent_2weights(X, weights_vanilla, bias, Y_, alpha_b, col1, col2)
        w1_van_lst.append(weights_vanilla.copy()[:, col1])
        w2_van_lst.append(weights_vanilla.copy()[:, col2])
    gd_traj = []
    for i in range(k+1):
        gd_traj.append((w1_van_lst[i][neuron], w2_van_lst[i][neuron]))

    weights_momentum = weights.copy()
    w1_mom_lst = []
    w2_mom_lst = []
    # adjust w1 w2 away from minimum
    weights_momentum[:, col1] -= width * 0.75
    weights_momentum[:, col2] -= width * 0.75
    w1_mom_lst.append(weights_momentum.copy()[:, col1])
    w2_mom_lst.append(weights_momentum.copy()[:, col2])
    # take k steps of momentum gradient descent
    Vw = np.zeros(weights.shape)
    for i in range(k):
        Vw = gradient_descent_momentum_2weights(X, weights_momentum, bias, Y_, alpha_c, col1, col2, Vw)
        w1_mom_lst.append(weights_momentum.copy()[:, col1])
        w2_mom_lst.append(weights_momentum.copy()[:, col2])
    mo_traj = []
    for i in range(k+1):
        mo_traj.append((w1_mom_lst[i][neuron], w2_mom_lst[i][neuron]))

    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc='upper left')
    plt.title("Contour Plot with Trajectories for the Cost Function of Neuron " + str(neuron))
    plt.savefig("Part 6(b)(c) Contour Plot")

    print "End of Part 6"
''' ##### End of Part 6 ##### '''
    
def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases (10 for digits)'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 

def check_num_images():
    total_training = 0
    total_test = 0
    for i in range(10):
        print "Number of digit "+str(i)+" training images: "+str(len(M['train'+str(i)]))
        total_training += len(M['train'+str(i)])
    print("Total Number of Training Images: "+str(total_training))
    for i in range(10):
        print "Number of digit "+str(i)+" test images: "+str(len(M['test'+str(i)]))
        total_test += len(M['test'+str(i)])
    print("Total Number of Test Images: "+str(total_test))
    
if __name__ == "__main__":
    #check_num_images()
    part1()
    part3()
    part4()
    part5()
    part6()
