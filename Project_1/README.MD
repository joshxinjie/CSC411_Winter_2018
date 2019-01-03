# Project 1: Face Recognition and Gender Classification with Regression

## Description
Build a system based on linear regression for face recognition and gender classification from scratch with just numpy. We will be using images of actors and actresses to train and test our system. These images were obtained from [FaceScrub](http://vintage.winklerbros.net/facescrub.html).

The main programming tasks of the project involved:

Task 1: Build a classifier to distinguish pictures of Alec Baldwin from Steve Carell
Accuracy on the train set: 100.0%
Accuracy of the test set: 95.0%

Task 2: Build a classifier to perform gender classification. We will train and test the system on images from 6 actors/actresses. We will then test classifier on 30 images of unknown actor/actresses:
Accuracy on the train set (6 known actors/actresses): 98.97%
Accuracy on the test set (6 known actors/actresses): 91.67%
Accuracy on the unknown actor/actresses test set: 85.56%

Task 3: Build a classifier to peform facial recognition for 6 actors/actresses.
Accuracy on the train set: 98.48%
Accuracy of the test set: 80.0%

The full report of the project can be found [here](https://github.com/joshxinjie/CSC411_Winter_2018/blob/master/Project_1/faces.pdf)

The detailed requirements for the project can be found at [link](https://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj1/).

## Files
faces.py - Main script for the project
faces.tex - Project Report LaTeX file
faces.pdf - Project Report
facescrub_actors.txt - Contains the download links for the actors' images
facescrub_actresses.txt - Contains the download links for the actresses' images
get_data.py - Downloads and pre-processes the images from the download links
rgb2gray.py - Contains helper function to convert RGB images to greyscale
subset_actors.txt - Contains list of actors' and actresses' names whose images we will work with in this project

## Installations
Anaconda
