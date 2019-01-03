# Project 2: Deep Neural Networks for Handwritten Digit and Face Recognition

## Description
Build several neural networks of varying depth, and use them for handwritten digit recognition and face recognition. Written in Python3.

The main programming tasks of the project are:

Task 1: Implement a single layer neural network with just numpy to perform handwritten digit recognition on the MNIST dataset. First, train the network with 'vanilla' gradient descent, then repeat the training of the network with momentum gradient descent. Optimize the network with cross-validation.

Results using 'vanilla' gradient descent
* Train Accuracy: 92.77%
* Validation Accuracy: 91.83%
* Test Accuracy:: 92.21%

Results using momentum gradient descent
* Train Accuracy: 93.81%
* Validation Accuracy: 91.93%
* Test Accuracy:: 92.43%

Task 2: Implement a single-hidden layer neural network with Pytorch. Using the same images of 6 actors and actresses from project 1, perform facial recognition. The images are obtained from [FaceScrub](http://vintage.winklerbros.net/facescrub.html)

Results:
* Train Accuracy: 100%
* Validation Accuracy: 75.0%
* Test Accuracy: 80.0%

Task 3: Use transfer learning with AlexNet to perform facial recognition on the 6 actors/actresses. Pytorch is used for this task as well. 

Results:
* Train Accuracy: 100%
* Validation Accuracy: 95.0%
* Test Accuracy: 93.3%

The full report on the project can be viewed [here](https://github.com/joshxinjie/CSC411_Winter_2018/blob/master/Project_2/deepnn.pdf)

The detailed requirements for the project can be found at [link](https://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj2/).

## Files
1. deepfaces.py - Script containing implementation of the single-hidden layer neural network and the transfer learning solution. The networks in this section are implemented with Pytorch. Script also contains code for performing facial recognition.
2. deepnn.pdf - Project Report.
3. deepnn.tex - Project Report LaTeX file.
4. digits.py - Script containing implementation of single-layer neural network. Network is implemented from scratch with just numpy. Script also contains code for performing handwritten digit recognition.
5. part5_bias.csv/part5_weights.csv - Contains saved weights and biases from trained network. Used in answering homework question.  

## Installations
Anaconda, Pytorch, urllib, hashlib, glob
