# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:51:24 2018

Using Python 3.6

@author: Xin Jie (Josh) Lee
"""

import glob as gl
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt
from numpy.linalg import norm

''' ########## Start Of Equations ########## '''
def cost_function(thetas, X, Y):
    return (1.0/(2.0*float(len(Y))))*np.sum((np.dot(X, thetas) - Y)**2)

def gradient(thetas, X, Y):
    return (1.0/(float(len(Y))))*np.dot(X.T, (np.dot(X, thetas) - Y))
''' ########## End Of Equations ########## '''

''' ########## Start Of Preprocessing Images Helper Functions ########## '''
def preprocess_images(name, gender_ind, training_set_size, validation_set_size, test_set_size):
    '''
    Returns training set, validation set and test set of images for given actor/actress
    gender_ind is 0 for male, 1 for female
    '''
    training_set = []
    validation_set = []
    test_set = []
    
    lastname = name.split()[1].lower().strip()
    
    if gender_ind == 0:
        images = gl.glob('cropped/male/' + lastname + '*')
        np.random.shuffle(images)
    elif gender_ind == 1:
        images = gl.glob('cropped/female/' + lastname + '*')
        np.random.shuffle(images)
    
    num_images = len(images)
    
    for i in range(num_images):
        image = sm.imread(images[i])
        # Flatten image to a vector
        flat_image = image.flatten()
        # Normalize image. Range of pixel intensity is between 0 to 255
        norm_flat_image = flat_image/255.0
        # Append constant 1 to the front of image array
        proc_image = np.insert(norm_flat_image, 0, 1)
        if i < training_set_size:
            # Add images to training set
            training_set.append(proc_image)
        elif (i >= training_set_size and i < training_set_size + validation_set_size):
            # Add 10 images to validation set
            validation_set.append(proc_image)
        elif (i >= training_set_size + validation_set_size and i < training_set_size + validation_set_size + test_set_size):
            # Add 10 images to test set
            test_set.append(proc_image)
    
    return training_set, validation_set, test_set

def preprocess_input_sets(people, training_set_size, validation_set_size, test_set_size):
    actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
    actresses =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
    i = 0
    training_set = []
    validation_set =[]
    test_set = []
    for act in people:
        if act in actors:
            training_set, validation_set, test_set = preprocess_images(act, 0, training_set_size, validation_set_size, test_set_size)
        elif act in actresses:
            training_set, validation_set, test_set = preprocess_images(act, 1, training_set_size, validation_set_size, test_set_size)
        if i == 0:
            training_input_all = training_set
            validation_input_all = validation_set
            test_input_all = test_set
        else:
            training_input_all = np.vstack((training_input_all, training_set))
            validation_input_all = np.vstack((validation_input_all, validation_set))
            test_input_all = np.vstack((test_input_all, test_set))
        i+=1
    return training_input_all, validation_input_all, test_input_all

def preprocess_labels_one(training_set_size, validation_set_size, test_set_size):
    '''
    Returns labels for training set and actual labels for validation and test sets
    for 1 individual. Person will be given label 1
    '''
    training_labels = np.ones((training_set_size,1))
    validation_actual_labels = np.ones((validation_set_size,1))
    test_actual_labels = np.ones((test_set_size,1))
    
    return training_labels, validation_actual_labels, test_actual_labels

def preprocess_labels_two(training_set_size, validation_set_size, test_set_size):
    '''
    Returns labels for training set and actual labels for validation and test sets
    for 2 class classification. 
    '''
    training_labels = np.ones((training_set_size*2,1))
    validation_actual_labels = np.ones((validation_set_size*2,1))
    test_actual_labels = np.ones((test_set_size*2,1))
    
    # First Person is given label 1, Second Person is given label -1
    training_labels[training_set_size:,0] = -1
    validation_actual_labels[validation_set_size:,0] = -1
    test_actual_labels[test_set_size:,0] = -1
    
    return training_labels, validation_actual_labels, test_actual_labels

def preprocess_labels_multi(num_ind, training_set_size, validation_set_size, test_set_size):
    '''
    Returns labels for training set and actual labels for validation and test sets
    for multiple class classification
    '''
    training_labels = np.zeros((training_set_size*num_ind,num_ind))
    validation_actual_labels = np.zeros((validation_set_size*num_ind,num_ind))
    test_actual_labels = np.zeros((test_set_size*num_ind,num_ind))
    
    for i in range(num_ind):
        training_labels[i*training_set_size:(i+1)*training_set_size,i] = 1
        validation_actual_labels[i*validation_set_size:(i+1)*validation_set_size,i] = 1
        test_actual_labels[i*test_set_size:(i+1)*test_set_size,i] = 1
    
    return training_labels, validation_actual_labels, test_actual_labels
''' ########## END Of Preprocessing Images Helper Functions ########## '''
  
''' ########## Start Of Gradient Descent ########## '''
def gradient_descent(multi, initial_thetas, learning_rate, training_inputs, training_labels, validation_inputs, validation_labels, max_iter):
    eps = 1e-5
    prev_thetas = initial_thetas - 10*eps
    thetas = initial_thetas.copy()
    i = 0
    while norm(thetas - prev_thetas) > eps and i < max_iter:
        prev_thetas = thetas.copy()
        thetas -= learning_rate*gradient(thetas, training_inputs, training_labels)
        i+=1
    if multi == 0: # 2 labels
        training_performance = performance(thetas, training_inputs, training_labels)
        validation_performance = performance(thetas, validation_inputs, validation_labels)
    elif multi == 1: # multi labels
        training_performance = performance_multi(thetas, training_inputs, training_labels)
        validation_performance = performance_multi(thetas, validation_inputs, validation_labels)
    training_cost = cost_function(thetas, training_inputs, training_labels)
    validation_cost = cost_function(thetas, validation_inputs, validation_labels)
    return thetas, training_performance, validation_performance, training_cost, validation_cost
''' ########## End Of Gradient Descent ########## '''

''' ########## Start Of Finite Difference ########## '''
def finite_difference(inputs, thetas, labels, h):
    ''' Returns average error per theta '''
    total_error = 0
    for i in range(thetas.shape[0]):
        for j in range(thetas.shape[1]):
            grad = gradient(thetas, inputs, labels)
            old_cost = cost_function(thetas, inputs, labels)
            new_thetas = thetas
            # Change one theta by h, hold all others constant
            new_thetas[i][j] += h
            new_cost = cost_function(new_thetas, inputs, labels)
            fin_diff = (new_cost - old_cost)/float(h)
            total_error += (fin_diff - grad[i][j])
    average_error = total_error/float(thetas.shape[0]*thetas.shape[1])
    return average_error
''' ########## End Of Finite Difference ########## '''

''' ########## Start Of Performance ########## '''
def performance(thetas, inputs, labels):
    ''' Returns accuracy of predicted output vs actual output
    for 2 class classification '''
    accuracy = 0
    predictions = np.dot(inputs, thetas)
    for i in range(len(predictions)):
        if predictions[i] > 0:
            # Prediction is closer to 1
            accuracy += (labels[i] == 1)
        elif predictions[i] < 0:
            # Prediction is closer to -1
            accuracy += (labels[i] == -1)
    return accuracy/float(len(labels))

def performance_multi(thetas, inputs, labels):
    ''' Returns accuracy of predicted output vs actual output
    for multiple class classification '''
    accuracy = 0
    predictions = np.dot(inputs, thetas)
    for i in range(len(predictions)):
        # If index of maximium element in each row (image) or predictions is equal to 
        # the index of 1 in the corresponding row of the label matrix
        if np.argmax(predictions[i]) == np.argmax(labels[i]):
            accuracy += 1
    return accuracy/float(len(labels))
''' ########## END Of Performance ########## '''

def part3():
    print("Running Part 3 ...")
    actors = ["Alec Baldwin", "Steve Carell"]
    
    training_set_size = 100
    validation_set_size = 10
    test_set_size = 10
    
    # Create inputs and labels
    training_input_all, validation_input_all, test_input_all = preprocess_input_sets(actors, training_set_size, validation_set_size, test_set_size)
    training_labels_all, validation_labels_all, test_labels_all = preprocess_labels_two(training_set_size, validation_set_size, test_set_size)
    
    # Initialize thetas. There are 1024 pixels plus 1 constant
    initial_thetas = np.vstack([0.0] for i in range(1025))
    
    # Set iterations for gradient descent
    num_iterations = [1000, 5000, 10000, 15000]
    
    for max_iter in num_iterations:
        learning_rates = np.arange(0.0005, 0.006, 0.0005)
        training_performances = []
        validation_performances = []
        training_costs = []
        validation_costs = []
        for learning_rate in learning_rates:
            final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input_all, training_labels_all, validation_input_all, validation_labels_all, max_iter)
            training_performances.append(training_performance)
            validation_performances.append(validation_performance)
            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
    
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(learning_rates, training_performances, label="Training Performance")
        ax1.plot(learning_rates, validation_performances, 'r-', label="Validation Performance")
        plt.xlabel("Learning Rate")
        plt.ylabel("Performance (% Accurate)")
        plt.title("Performance for Steve Carell and Alec Baldwin (Max "+str(max_iter)+" iterations)")
        plt.legend(loc=0)
        plt.savefig("Part3_Learning_Rate_Performances"+str(max_iter)+"iterations")
    
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(learning_rates, training_costs, label="Training Costs")
        ax2.plot(learning_rates, validation_costs, 'r-', label="Validation Costs")
        plt.xlabel("Learning Rate")
        plt.ylabel("Cost")
        plt.title("Costs for Steve Carell and Alec Baldwin (Max "+str(max_iter)+" iterations)")
        plt.legend(loc=0)
        plt.savefig("Part3_Learning_Rate_Cost"+str(max_iter)+"iterations")
    
    # Pick 0.005 for alpha and a maximum of 10,000 iterations for final selection
    learning_rate = 0.005
    max_iter = 10000
    final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input_all, training_labels_all, validation_input_all, validation_labels_all, max_iter)
    
    print("Results for chosen learning rate of 0.005 and maximum number of iterations of 10,000")
    print("Training Performance:",training_performance[0]*100, "%")
    print("Validation Performance:", validation_performance[0]*100, "%")
    print("Training Cost:",training_cost)
    print("Validation Cost:", validation_cost)
    
    print("Part 3 is completed!\n")
    
def part4a():
    print("Running Part 4 (a) ...")
    actors = ["Alec Baldwin", "Steve Carell"]
    
    '''### Plotting thetas for full training set ###'''
    for act in actors:
        training_set_size = 100
        validation_set_size = 10
        test_set_size = 10
        
        actor = []
        actor.append(act)
    
        # Set up input images for training set, validation set and test set
        training_input, validation_input, test_input = preprocess_input_sets(actor, training_set_size, validation_set_size, test_set_size)
        training_labels, validation_labels, test_labels = preprocess_labels_one(training_set_size, validation_set_size, test_set_size)
        training_input = np.vstack([training_input])
        validation_input = np.vstack([validation_input])
        test_input = np.vstack([test_input])
        
        # Initialize thetas. There are 1024 pixels plus 1 bias term
        initial_thetas = np.vstack([0.0] for i in range(1025))

        # Initialize learning rate
        learning_rate = 0.005
        max_iter = 10000
        final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input, training_labels, validation_input, validation_labels, max_iter)
        # Remove first theta that is for bias
        plot_thetas_100 = final_thetas[1:]
        plot_thetas_100.shape = (32, 32)
        sm.imsave("Part4a_100_training_set_"+str(act.split()[1])+".jpg", plot_thetas_100)
    
    '''### Plotting thetas for training set of size 2 ###'''
    for act in actors:
        training_set_size = 2
        validation_set_size = 2
        test_set_size = 2
        
        actor = []
        actor.append(act)
    
        # Set up input images for training set, validation set and test set
        training_input, validation_input, test_input = preprocess_input_sets(actor, training_set_size, validation_set_size, test_set_size)
        training_labels, validation_labels, test_labels = preprocess_labels_one(training_set_size, validation_set_size, test_set_size)
        training_input = np.vstack([training_input])
        validation_input = np.vstack([validation_input])
        test_input = np.vstack([test_input])
        # Initialize thetas. There are 1024 pixels plus 1 bias term
        initial_thetas = np.vstack([0.0] for i in range(1025))
    
        final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input, training_labels, validation_input, validation_labels, max_iter)
        # Remove first theta that is for bias
        plot_thetas_2 = final_thetas[1:]
        plot_thetas_2.shape = (32, 32)
        sm.imsave("Part4a_2_training_set_"+str(act.split()[1])+".jpg", plot_thetas_2)
    
    print("Part 4 (a) is completed!\n")

def part4b():
    print("Running Part 4 (b) ...")
    actors = ["Alec Baldwin", "Steve Carell"]
    
    ''' ### Plotting thetas for 50 iterations ### '''
    for act in actors:
        training_set_size = 100
        validation_set_size = 10
        test_set_size = 10
        
        actor = []
        actor.append(act)
        
        # Set up input images for training set, validation set and test set
        training_input, validation_input, test_input = preprocess_input_sets(actor, training_set_size, validation_set_size, test_set_size)
        training_labels, validation_labels, test_labels = preprocess_labels_one(training_set_size, validation_set_size, test_set_size)
        training_input = np.vstack([training_input])
        validation_input = np.vstack([validation_input])
        test_input = np.vstack([test_input])
        # Initialize thetas. There are 1024 pixels plus 1 bias term
        initial_thetas = np.vstack([0.0] for i in range(1025))

        # Initialize learning rate
        learning_rate = 0.005
        max_iter = 50
        final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input, training_labels, validation_input, validation_labels, max_iter)
        # Remove first theta that is for bias
        plot_thetas_100iter = final_thetas[1:]
        plot_thetas_100iter.shape = (32, 32)
        sm.imsave("Part4b_"+str(max_iter)+"_iter_"+str(act.split()[1])+".jpg", plot_thetas_100iter)
    
    ''' ### Plotting thetas for 20,000 iterations ### '''
    for act in actors:
        training_set_size = 100
        validation_set_size = 10
        test_set_size = 10
        
        actor = []
        actor.append(act)
    
        # Set up input images for training set, validation set and test set
        training_input, validation_input, test_input = preprocess_input_sets(actor, training_set_size, validation_set_size, test_set_size)
        training_labels, validation_labels, test_labels = preprocess_labels_one(training_set_size, validation_set_size, test_set_size)
        training_input = np.vstack([training_input])
        validation_input = np.vstack([validation_input])
        test_input = np.vstack([test_input])
        # Initialize thetas. There are 1024 pixels plus 1 bias term
        initial_thetas = np.vstack([0.0] for i in range(1025))

        max_iter = 20000
    
        final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_input, training_labels, validation_input, validation_labels, max_iter)
        # Remove first theta that is for bias
        plot_thetas_20000iter = final_thetas[1:]
        plot_thetas_20000iter.shape = (32, 32)
        sm.imsave("Part4b_"+str(max_iter)+"_iter_"+str(act.split()[1])+".jpg", plot_thetas_20000iter)
    
    print("Part 4 (b) is completed!\n")
    
def part5():
    print("Running Part 5 ...")
    people = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    people_other = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
    
    training_set_sizes = range(5, 70, 5)
    validation_set_size = 10
    test_set_size = 10
    
    training_set_size_array = []
    training_performances = []
    validation_performances = []
    unknown_performances = []
    
    other_training_set_size = 1
    other_validation_set_size = 1
    other_test_set_size = 30
    
    max_iter = 10000
    
    for training_set_size in training_set_sizes:
        training_inputs, validation_inputs, test_inputs = preprocess_input_sets(people, training_set_size, validation_set_size, test_set_size)
        # Female labels are 1, male labels are -1
        training_labels, validation_labels, test_labels = preprocess_labels_two(3*training_set_size, 3*validation_set_size, 3*test_set_size)
        initial_thetas = np.vstack([0.0] for i in range(1025))
        learning_rate = 0.005
        final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(0, initial_thetas, learning_rate, training_inputs, training_labels, validation_inputs, validation_labels, max_iter)
        training_performances.append(training_performance)
        validation_performances.append(validation_performance)
        
        # Generating test sets and test labels for 6 actors not included in act
        other_training_inputs, other_validation_inputs, other_test_inputs = preprocess_input_sets(people_other, other_training_set_size, other_validation_set_size, other_test_set_size)
        other_training_labels, other_validation_labels, other_test_labels = preprocess_labels_two(3*other_training_set_size, 3*other_validation_set_size, 3*other_test_set_size)
        unknown_performances.append(performance(final_thetas, other_test_inputs, other_test_labels))
        training_set_size_array.append(training_set_size)
    
    unknown_performances = np.vstack(unknown_performances).flatten()
    
    print("Final results of gender classification for known actors/actresses")
    print("Training Performance:",training_performances[-1][0]*100, "%")
    print("Validation Performance:", validation_performances[-1][0]*100, "%\n")
    
    print("Final result of gender classification for unknown actors/actresses")
    print("Classification Performance:",unknown_performances[-1]*100, "%")
    
    # Plotiing performance on training and validation sets for the 6 known actors
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(training_set_size_array, training_performances, label="Training Performance")
    ax1.plot(training_set_size_array, validation_performances, 'r-', label="Validation Performance")
    plt.xlabel("Size of Training Set")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Performance on Training and Validation Sets for 6 Known Actors")
    plt.legend(loc=0)
    plt.savefig("Part6_Training_validation")
    
    # Plotting performance on test sets for 6 unknown actors
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(training_set_size_array, unknown_performances, label="Test Performance")
    plt.xlabel("Size of Training Set")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Performance on Test Sets for 6 Unknown Actors")
    plt.legend(loc=0)
    plt.savefig("Part6_Unknown_Test")
    
    print("Part 5 is completed!\n")

def part6():
    print("Running Part 6 ...")
    people = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_set_size = 66
    validation_set_size = 10
    test_set_size = 10
    
    # Create input matrix
    training_inputs, validation_inputs, test_inputs = preprocess_input_sets(people, training_set_size, validation_set_size, test_set_size)
    # Create labels
    training_labels, validation_labels, test_labels = preprocess_labels_multi(6, training_set_size, validation_set_size, test_set_size)
    # Initialize thetas
    thetas = np.vstack([0.0,0.0,0.0,0.0,0.0,0.0] for i in range(1025))
    
    error = finite_difference(training_inputs, thetas, training_labels, 0.0001)
    print("Average error between finite difference and gradient function using h = 0.0001 is:", error)
    
    print("Part 6 is completed!\n")
    
def part7():
    print("Running Part 7 ...")
    people = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_set_size = 66
    validation_set_size = 10
    test_set_size = 10
    
    # Create input matrix
    training_inputs, validation_inputs, test_inputs = preprocess_input_sets(people, training_set_size, validation_set_size, test_set_size)
    # Create labels
    training_labels, validation_labels, test_labels = preprocess_labels_multi(6, training_set_size, validation_set_size, test_set_size)
    
    initial_thetas = np.vstack([0.0,0.0,0.0,0.0,0.0,0.0] for i in range(1025))
    num_iterations = [1000, 5000, 10000, 15000]
    training_performances = []
    validation_performances = []
    training_costs = []
    validation_costs = []
    for max_iter in num_iterations:
        learning_rates = np.arange(0.0005, 0.006, 0.0005)
        training_performances = []
        validation_performances = []
        training_costs = []
        validation_costs = []
        for learning_rate in learning_rates:
            final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(1, initial_thetas, learning_rate, training_inputs, training_labels, validation_inputs, validation_labels, max_iter)
            training_performances.append(training_performance)
            validation_performances.append(validation_performance)
            training_costs.append(training_cost)
            validation_costs.append(validation_cost)
    
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(learning_rates, training_performances, label="Training Performance")
        ax1.plot(learning_rates, validation_performances, 'r-', label="Validation Performance")
        plt.xlabel("Learning Rate")
        plt.ylabel("Performance (% Accurate)")
        plt.title("Learning Rate vs Performance (Max "+str(max_iter)+" iterations)")
        plt.legend(loc=0)
        plt.savefig("Part7_Learning_Rate_Performances"+str(max_iter)+"iterations")
    
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(learning_rates, training_costs, label="Training Costs")
        ax2.plot(learning_rates, validation_costs, 'r-', label="Validation Costs")
        plt.xlabel("Learning Rate")
        plt.ylabel("Cost")
        plt.title("Learning Rate vs Costs (Max "+str(max_iter)+" iterations)")
        plt.legend(loc=0)
        plt.savefig("Part7_Learning_Rate_Cost"+str(max_iter)+"iterations")
    
    # Initialize thetas
    initial_thetas = np.vstack([0.0,0.0,0.0,0.0,0.0,0.0] for i in range(1025))
    # Set learning rate 
    learning_rate = 0.005
    max_iter = 10000
    # Train Model
    final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(1, initial_thetas, learning_rate, training_inputs, training_labels, validation_inputs, validation_labels, max_iter)
    
    print("Results for chosen learning rate of "+str(learning_rate)+" and maximum number of iterations of "+str(max_iter))
    print("Final training set performance:", training_performance*100, "%")
    print("Final validation set performance:", validation_performance*100, "%")
    print("Final training cost:", training_cost)
    print("Final validation cost:", validation_cost)
    
    print("Part 7 is completed!\n")

def part8():
    print("Running Part 8 ...")
    people = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_set_size = 66
    validation_set_size = 10
    test_set_size = 10
    
    # Create input matrix
    training_inputs, validation_inputs, test_inputs = preprocess_input_sets(people, training_set_size, validation_set_size, test_set_size)
    # Create labels
    training_labels, validation_labels, test_labels = preprocess_labels_multi(6, training_set_size, validation_set_size, test_set_size)
    # Initialize thetas
    initial_thetas = np.vstack([0.0,0.0,0.0,0.0,0.0,0.0] for i in range(1025))
    # Set learning rate
    learning_rate = 0.005
    max_iter = 10000
    # Train Model
    final_thetas, training_performance, validation_performance, training_cost, validation_cost = gradient_descent(1, initial_thetas, learning_rate, training_inputs, training_labels, validation_inputs, validation_labels, max_iter)
    plot_thetas = final_thetas[1:,:]
    plot_thetas = plot_thetas.T
    for i in range(plot_thetas.shape[0]):
        theta_image = plot_thetas[i]
        theta_image.shape = (32, 32)
        sm.imsave("Part8_person_"+str(i)+".jpg", theta_image)
    
    print("Part 8 is completed!")

if __name__ == '__main__':
    part3()
    part4a()
    part4b()
    part5()
    part6()
    part7()
    part8()
