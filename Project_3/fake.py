# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:58:57 2018
Written in Python 3.6
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import operator
import _pickle as cPickle

RAND_SEED = 7

P4_WORDS_CSV = "Part4_words.csv"
P4_WEIGHTS_CSV = "Part4_weights.csv"

"""===================================HELPER==================================="""
class LogisticRegression(torch.nn.Module):
    def __init__(self, dim_x):
        """
        In the constructor we instantiate nn.Linear module
        """       
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(dim_x, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

def setup():
    """
    Creates training, validation and test data in a 70/15/15 split.
    """
    np.random.seed(RAND_SEED)
    real_headlines = np.array([a.split("\n")[0] for a in open("clean_real.txt").readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open("clean_fake.txt").readlines()])
    real_rand = np.random.permutation(len(real_headlines))
    fake_rand = np.random.permutation(len(fake_headlines))
    real_train = real_headlines[real_rand[:int(round(0.7*len(real_rand)))]]
    real_valid = real_headlines[real_rand[int(round(0.7*len(real_rand))):int(round(0.85*len(real_rand)))]]
    real_test = real_headlines[real_rand[int(round(0.85*len(real_rand))):]]
    fake_train = fake_headlines[fake_rand[:int(round(0.7*len(fake_rand)))]]
    fake_valid = fake_headlines[fake_rand[int(round(0.7*len(fake_rand))):int(round(0.85*len(fake_rand)))]]
    fake_test = fake_headlines[fake_rand[int(round(0.85*len(fake_rand))):]]
    
    return real_train, real_valid, real_test, fake_train, fake_valid, fake_test

def final_setup_log(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, all_keywords):
    """
    Return train, valid, test inputs with indicator variables in place of word
    Creates train, valid, test labels
    For Logistic Regression
    Has bias unit added to every input
    """
    # Create labels
    Y_train = np.zeros(len(real_train)+len(fake_train))
    Y_valid = np.zeros(len(real_valid)+len(fake_valid))
    Y_test = np.zeros(len(real_test)+len(fake_test))
    # Real is 1, fake is 0
    Y_train[:len(real_train)] = 1
    Y_valid[:len(real_valid)] = 1
    Y_test[:len(real_test)] = 1
    
    Y_train = Y_train.reshape(len(real_train)+len(fake_train), 1)
    Y_valid = Y_valid.reshape(len(real_valid)+len(fake_valid), 1)
    Y_test = Y_test.reshape(len(real_test)+len(fake_test), 1)
    
    
    # Create X_train input
    i = 0
    for headline in real_train:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    # First unit of x_curr is bias, so add 1 to index
                    x_curr[index+1] = 1
        if i == 0:
            X_train = np.vstack([x_curr])
        else:
            X_train = np.vstack((X_train, x_curr))
        i += 1
    for headline in fake_train:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index+1] = 1
        X_train = np.vstack((X_train, x_curr))
    
    # Create X_valid input
    i = 0
    for headline in real_valid:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index+1] = 1
        if i == 0:
            X_valid = np.vstack([x_curr])
        else:
            X_valid = np.vstack((X_valid, x_curr))
        i += 1
    for headline in fake_valid:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index+1] = 1
        X_valid = np.vstack((X_valid, x_curr))
    
    # Create X_test input
    i = 0
    for headline in real_test:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index+1] = 1
        if i == 0:
            X_test = np.vstack([x_curr])
        else:
            X_test = np.vstack((X_test, x_curr))
        i += 1
    for headline in fake_test:
        x_curr = np.zeros(len(all_keywords)+1)
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index+1] = 1
        X_test = np.vstack((X_test, x_curr))
    
    # add bias
    X_train[:,0] = 1
    X_valid[:,0] = 1
    X_test[:,0] = 1
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def final_setup(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, all_keywords):
    """
    Return train, valid, test inputs with indicator variables in place of word
    Creates train, valid, test labels
    For Decision Tree
    No bias unit in input
    """
    # Create labels
    Y_train = np.zeros(len(real_train)+len(fake_train))
    Y_valid = np.zeros(len(real_valid)+len(fake_valid))
    Y_test = np.zeros(len(real_test)+len(fake_test))
    # Real is 1, fake is 0
    Y_train[:len(real_train)] = 1
    Y_valid[:len(real_valid)] = 1
    Y_test[:len(real_test)] = 1
    
    Y_train = Y_train.reshape(len(real_train)+len(fake_train), 1)
    Y_valid = Y_valid.reshape(len(real_valid)+len(fake_valid), 1)
    Y_test = Y_test.reshape(len(real_test)+len(fake_test), 1)
    
    
    # Create X_train input
    i = 0
    for headline in real_train:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    # First unit of x_curr is bias, so add 1 to index
                    x_curr[index] = 1
        if i == 0:
            X_train = np.vstack([x_curr])
        else:
            X_train = np.vstack((X_train, x_curr))
        i += 1
    for headline in fake_train:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index] = 1
        X_train = np.vstack((X_train, x_curr))
    
    # Create X_valid input
    i = 0
    for headline in real_valid:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index] = 1
        if i == 0:
            X_valid = np.vstack([x_curr])
        else:
            X_valid = np.vstack((X_valid, x_curr))
        i += 1
    for headline in fake_valid:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index] = 1
        X_valid = np.vstack((X_valid, x_curr))
    
    # Create X_test input
    i = 0
    for headline in real_test:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index] = 1
        if i == 0:
            X_test = np.vstack([x_curr])
        else:
            X_test = np.vstack((X_test, x_curr))
        i += 1
    for headline in fake_test:
        x_curr = np.zeros(len(all_keywords))
        words = headline.split()
        counted = []
        for word in words:
            # If word in headline has not been counted
            if word not in counted:
                # If word in training keywords
                if word in all_keywords:
                    index = all_keywords.index(word)
                    x_curr[index] = 1
        X_test = np.vstack((X_test, x_curr))
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
 
def num_headlines(real_headlines, fake_headlines):
    """
    Return the number of real and fake headlines.
    """    
    return float(len(real_headlines)), float(len(fake_headlines))

def p_wi_class(headlines, all_words, m, p):
    """
    Returns a dictionary of P(w_i=1 | class) for all words.
    P(w_i=1 | class) = (number of 'class' headlines that contain word w_i + m*p)/
                        (total number of 'class' headlines + m)
    For Naive Bayes Model
    """
    class_dict = {}
    for word in all_words:
        class_dict[word] = 0.0
    for headline in headlines:
        # Count any repeated words in a headline once
        counted = []
        words = headline.split()
        for word in words:
            if word not in counted:
                class_dict[word] += 1.0
                counted.append(word)
    for word in class_dict:
        class_dict[word] += m*p
        class_dict[word] /= (len(headlines) + m)
    return class_dict

def p_not_wi_class(headlines, all_words, m, p):
    """
    Returns a dictionary of P(w_i=0 | class) for all words.
    P(w_i=0 | class) = (number of 'class' headlines that do not contain word w_i)/
                        (total number of 'class' headlines)
                     = 1 - P(w_i=1 | class)
    For Naive Bayes
    """
    class_dict = {}
    for word in all_words:
        class_dict[word] = 0.0
    for headline in headlines:
        counted = []
        words = headline.split()
        for word in words:
            if word not in counted:
                class_dict[word] += 1.0
                counted.append(word)
    final_class_dict = {}
    for word in class_dict:
        class_dict[word] += m*p
        class_dict[word] /= (len(headlines) + m)
        final_class_dict[word] = 1.0 - class_dict[word]
    return final_class_dict

def p_class_wi(real_likelihood, fake_likelihood, num_real, num_fake):
    """
    Returns a dictionary of posterior distributions P(class | w_i = 1) 
    = P(w_i = 1 | class)P(class)/P(w_i = 1) where P(w_i = 1) = 
    P(w_i = 1 | class) + P(w_i = 1 | not class) and P(class) =
    (number of 'class' news headlines) / (total number of news headlines)
    For Naive Bayes Model
    """
    real_posterior = {}
    fake_posterior = {}
    real_prior = float(num_real/(num_real+num_fake))
    fake_prior = float(num_fake/(num_real+num_fake))
    # Build dictionary of real news posterior probabilities
    for word in real_likelihood:
        real_posterior[word] = (real_likelihood[word]*real_prior)/(real_likelihood[word] + fake_likelihood[word])
    # Build dictionary of fake news posterior probabilities
    for word in fake_likelihood:
        fake_posterior[word] = (fake_likelihood[word]*fake_prior)/(real_likelihood[word] + fake_likelihood[word])
    return real_posterior, fake_posterior

def all_words(real_headlines, fake_headlines):
    """
    Return a list of all unique words in both real and fake headlines
    """
    unique_words = []
    for headline in real_headlines:
        words = headline.split()
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    for headline in fake_headlines:
        words = headline.split()
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words

def classify(headline, num_real, num_fake, real_keywords, fake_keywords):
    """
    Classify a single headline as real news or fake news.
    Computes P(class | w_1, ... , w_n) which is proportional to sum of log(P(w_i | class))
    for all words w_i and log(P(class))  
    For Naive Bayes Model
    """
    real_sentiment = 0.0
    fake_sentiment = 0.0
    words = headline.split()
    for word in words:
        if word in real_keywords:
            real_sentiment += np.log(real_keywords[word])
        if word in fake_keywords:
            fake_sentiment += np.log(fake_keywords[word])
    # Add log(P(class)) to both sentiments
    real_sentiment += np.log(num_real/(num_real+num_fake))
    fake_sentiment += np.log(num_fake/(num_real+num_fake))
    if np.exp(real_sentiment) > np.exp(fake_sentiment):
        return 'real'
    else:
        return 'fake'

def performance(real_headlines, fake_headlines, real_keywords, fake_keywords):
    """
    Returns accuracy of predictions for Naive Bayes Model
    """
    real_correct = 0.0
    fake_correct = 0.0
    num_real = float(len(real_headlines))
    num_fake = float(len(fake_headlines))
    for headline in real_headlines:
        sentiment = classify(headline, num_real, num_fake, real_keywords, fake_keywords)
        if sentiment == "real":
            real_correct += 1.0
    for headline in fake_headlines:
        sentiment = classify(headline, num_real, num_fake, real_keywords, fake_keywords)
        if sentiment == "fake":
            fake_correct += 1.0
    perf = (real_correct + fake_correct)/(num_real + num_fake)
    real_perf = real_correct/num_real
    fake_perf = fake_correct/num_fake
    return perf, real_perf, fake_perf

def accuracy_log(Y_pred, Y_act):
    """
    Returns accuracy of predictions for Logistic Regression Model
    """
    accuracy = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] > 0.5 and Y_act[i] == 1:
            accuracy += 1
        elif Y_pred[i] < 0.5 and Y_act[i] == 0:
            accuracy += 1
    return accuracy/float(len(Y_pred))

def accuracy(Y_pred, Y_act):
    """
    Returns Accuracy of Predictions For Decision Tree Model
    For Part 7
    """
    accuracy = 0.0
    for i in range(len(Y_pred)):
        accuracy += (Y_pred[i] == Y_act[i])
    return (accuracy/float(len(Y_pred)))[0]

def top_keywords(real_keywords, fake_keywords, num_keywords):
    """
    Returns 'num_keywords' words with the highest posterior probabilities
    For Part 3
    """
    sorted_real_keywords = sorted(real_keywords.items(), key=operator.itemgetter(1), reverse=True)
    sorted_fake_keywords = sorted(fake_keywords.items(), key=operator.itemgetter(1), reverse=True)
    top_real_keywords = []
    top_fake_keywords = []
    for i in range(num_keywords):
        top_real_keywords.append(sorted_real_keywords[i])
        top_fake_keywords.append(sorted_fake_keywords[i])
    
    return top_real_keywords, top_fake_keywords

def top_keywords_log(keywords, num_keywords):
    """
    Returns the 'num_keywords' words with the highest thetas/weights and the
    'num_keywords' words with the lowest thetas/weights
    For Logistic Regression (Part 6)
    """
    sorted_top_keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)
    sorted_last_keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=False)
    top10_keywords = []
    last10_keywords = []
    for i in range(num_keywords):
        top10_keywords.append(sorted_top_keywords[i])
        last10_keywords.append(sorted_last_keywords[i])
    
    return top10_keywords, last10_keywords

def remove_stopwords(real_keywords, fake_keywords):
    """
    Remove stopwords from the dictionaries for real_keywords and fake_keywords
    For Part 3
    """
    for word in ENGLISH_STOP_WORDS:
        if word in real_keywords:
            real_keywords.pop(word)
        if word in fake_keywords:
            fake_keywords.pop(word)
    return real_keywords, fake_keywords

def remove_stopwords_log(keywords):
    """
    Remove stopwords from the dictionaries for keywords
    For Part 6
    """
    for word in ENGLISH_STOP_WORDS:
        if word in keywords:
            keywords.pop(word)
    return keywords

def remove_stopwords_part7(words):
    """
    Remove stop words from a list of unique words.
    For part 7
    """
    for word in words:
        if word in ENGLISH_STOP_WORDS:
            words.remove(word)
    return words
"""===================================HELPER==================================="""

"""===================================PART1==================================="""
def part1():
    print("Running Part 1 ...")
    print("\n")
    real_headlines = np.array([a.split("\n")[0] for a in open("clean_real.txt").readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open("clean_fake.txt").readlines()])
    all_keywords = all_words(real_headlines, fake_headlines)
    m = 0
    p = 0
    real_keywords = p_wi_class(real_headlines, all_keywords, m, p)
    fake_keywords = p_wi_class(fake_headlines, all_keywords, m, p)
    num_keywords = 3
    
    top_real_keywords, top_fake_keywords = top_keywords(real_keywords, fake_keywords, num_keywords)
    print("Top "+str(num_keywords)+" useful keywords for real headlines: ")
    for i in range(num_keywords):
        print("Word: "+str(top_real_keywords[i][0])+", Likelihood: "+str(top_real_keywords[i][1]))
    print("\n")
    print("Top "+str(num_keywords)+" useful keywords for fake headlines: ")
    for i in range(num_keywords):
        print("Word: "+str(top_fake_keywords[i][0])+", Likelihood: "+str(top_fake_keywords[i][1]))
    print("\n")
    print("End of Part 1")
    print("\n")
"""===================================PART1==================================="""

"""===================================PART2==================================="""
def part2():
    print("Running Part 2 ...")
    print("\n")
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
    all_keywords = all_words(real_train, fake_train)
    
    #ms = np.arange(1, 6, 1)
    #ps = np.arange(0.001, 1, 0.001)
    ms = [1] # Best
    ps = [0.312] # Best
    best_valid_perf = 0.0
    best_valid_real_perf = 0.0
    best_valid_fake_perf = 0.0
    best_train_perf = 0.0
    best_train_real_perf = 0.0
    best_train_fake_perf = 0.0
    best_m = 0.0
    best_p = 0.0
    
    
    for m in ms:
        for p in ps:
            print("Running model for m: "+str(m)+" and p: "+str(p))
            
            '''
            # Using likelihood only
            # Get real news and fake news keywords from training data only
            real_likelihood = p_wi_class(real_train, all_keywords, m, p)
            fake_likelihood = p_wi_class(fake_train, all_keywords, m, p)            
            train_perf, train_real_perf, train_fake_perf = performance(real_train, fake_train, real_likelihood, fake_likelihood)
            valid_perf, valid_real_perf, valid_fake_perf = performance(real_valid, fake_valid, real_likelihood, fake_likelihood)
            '''
            
            # Using posterior
            real_likelihood = p_wi_class(real_train, all_keywords, m, p)
            fake_likelihood = p_wi_class(fake_train, all_keywords, m, p)
            num_real, num_fake = num_headlines(real_train, fake_train)
            real_posterior, fake_posterior = p_class_wi(real_likelihood, fake_likelihood, num_real, num_fake)
            train_perf, train_real_perf, train_fake_perf = performance(real_train, fake_train, real_posterior, fake_posterior)
            valid_perf, valid_real_perf, valid_fake_perf = performance(real_valid, fake_valid, real_posterior, fake_posterior)
            
            if valid_perf > best_valid_perf:
                best_valid_perf = valid_perf
                best_train_perf = train_perf
                best_train_real_perf = train_real_perf
                best_train_fake_perf = train_fake_perf
                best_valid_real_perf = valid_real_perf
                best_valid_fake_perf = valid_fake_perf
                best_m = m
                best_p = p
                print("Results For Current Best Model")
                print("Current Best Training Peformance: " + str(best_train_perf*100) + "%")
                print("Current Best Training Real Peformance: " + str(best_train_real_perf*100) + "%")
                print("Current Best Training Fake Peformance: " + str(best_train_fake_perf*100) + "%")
                print("Current Best Validation Peformance: " + str(best_valid_perf*100) + "%")
                print("Current Best Validation Real Peformance: " + str(best_valid_real_perf*100) + "%")
                print("Current Best Validation Fake Peformance: " + str(best_valid_fake_perf*100) + "%")
                print("Current Best m: " + str(best_m))
                print("Current Best p: " + str(best_p))
    test_perf, test_real_perf, test_fake_perf = performance(real_test, fake_test, real_posterior, fake_posterior)
    print("\n")
    print("Results For Best Model")
    print("Best Training Peformance: " + str(best_train_perf*100) + "%")
    print("Best Training Real News Peformance: " + str(best_train_real_perf*100) + "%")
    print("Best Training Fake News Peformance: " + str(best_train_fake_perf*100) + "%")
    print("Best Validation Peformance: " + str(best_valid_perf*100) + "%")
    print("Best Validation Real News Peformance: " + str(best_valid_real_perf*100) + "%")
    print("Best Validation Fake News Peformance: " + str(best_valid_fake_perf*100) + "%")
    print("Best Test Peformance: " + str(test_perf*100) + "%")
    print("Best Test Real News Peformance: " + str(test_real_perf*100) + "%")
    print("Best Test Fake News Peformance: " + str(test_fake_perf*100) + "%")
    print("Best m: " + str(best_m))
    print("Best p: " + str(best_p))
    print("\n")
    print("End of Part 2")
    print("\n")
"""===================================PART2==================================="""

"""===================================PART3==================================="""
def part3a():
    """
    For words whose presence strongly predict a news is real, we compare the words
    with the highest posterior probability P(class | w_i = 1) = 
    P(w_i = 1 | class)P(class)/P(w_i = 1)
    
    For words whose absence strongly predict a news is real, we compare the words
    with the highest posterior probability P(class | w_i = 0) = 
    P(w_i = 0 | class)P(class)/P(w_i = 0)
    
    """
    print("Running Part 3(a) ...")
    print("\n")
    real_headlines = np.array([a.split("\n")[0] for a in open("clean_real.txt").readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open("clean_fake.txt").readlines()])
    all_keywords = all_words(real_headlines, fake_headlines)
    m = 1
    p = 0.312
    real_likelihood = p_wi_class(real_headlines, all_keywords, m, p)
    fake_likelihood = p_wi_class(fake_headlines, all_keywords, m, p)
    num_real, num_fake = num_headlines(real_headlines, fake_headlines)
    num_keywords = 10
    
    real_posterior, fake_posterior = p_class_wi(real_likelihood, fake_likelihood, num_real, num_fake)
    
    top_real_keywords, top_fake_keywords = top_keywords(real_posterior, fake_posterior, num_keywords)
    print("Top "+str(num_keywords)+" keywords whose presence strongly predict a news is real: ")
    for i in range(num_keywords):
        print("Word: "+str(top_real_keywords[i][0])+", Posterior: "+str(top_real_keywords[i][1]))
    print("\n")
    print("Top "+str(num_keywords)+" keywords whose presence strongly predict a news is fake: ")
    for i in range(num_keywords):
        print("Word: "+str(top_fake_keywords[i][0])+", Posterior: "+str(top_fake_keywords[i][1]))
    print("\n")
    print("-------------------------------------------------------")
    print("\n")
    real_absence_likelihood = p_not_wi_class(real_headlines, all_keywords, m, p)
    fake_absence_likelihood = p_not_wi_class(fake_headlines, all_keywords, m, p)
    
    real_absence_posterior, fake_absence_posterior = p_class_wi(real_absence_likelihood, fake_absence_likelihood, num_real, num_fake)
    
    top_real_absence, top_fake_absence = top_keywords(real_absence_posterior, fake_absence_posterior, num_keywords)
    print("Top "+str(num_keywords)+" keywords whose absence strongly predict a news is real: ")
    for i in range(num_keywords):
        print("Word: "+str(top_real_absence[i][0])+", Posterior: "+str(top_real_absence[i][1]))
    print("\n")
    print("Top "+str(num_keywords)+" keywords whose absence strongly predict a news is fake: ")
    for i in range(num_keywords):
        print("Word: "+str(top_fake_absence[i][0])+", Posterior: "+str(top_fake_absence[i][1]))
    print("\n")
    print("End of Part 3(a)")
    print("\n")

def part3b():
    print("Running Part 3(b) ...")
    print("\n")
    real_headlines = np.array([a.split("\n")[0] for a in open("clean_real.txt").readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open("clean_fake.txt").readlines()])
    num_real, num_fake = num_headlines(real_headlines, fake_headlines)
    all_keywords = all_words(real_headlines, fake_headlines)
    m = 1
    p = 0.312
    real_likelihood = p_wi_class(real_headlines, all_keywords, m, p)
    fake_likelihood = p_wi_class(fake_headlines, all_keywords, m, p)
    num_keywords = 10
    
    real_posterior, fake_posterior = p_class_wi(real_likelihood, fake_likelihood, num_real, num_fake)
    
    real_keywords, fake_keywords = remove_stopwords(real_posterior, fake_posterior)
    
    print("After Removing Stopwords")
    
    top_real_keywords, top_fake_keywords = top_keywords(real_keywords, fake_keywords, num_keywords)
    print("Top "+str(num_keywords)+" keywords whose presence strongly predict a news is real: ")
    for i in range(num_keywords):
        print("Word: "+str(top_real_keywords[i][0])+", Posterior: "+str(top_real_keywords[i][1]))
    print("\n")
    print("Top "+str(num_keywords)+" keywords whose presence strongly predict a news is fake: ")
    for i in range(num_keywords):
        print("Word: "+str(top_fake_keywords[i][0])+", Posterior: "+str(top_fake_keywords[i][1]))
    print("\n")
    print("End of Part 3(b)")
    print("\n")
"""===================================PART3==================================="""

"""===================================PART4==================================="""
def part4_best_model():
    """
    Grid search for best logistic regression model with L2 regularization
    """
    torch.manual_seed(RAND_SEED)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
    train_keywords = all_words(real_train, fake_train)
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup_log(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
    dim_x = X_train.shape[1]
    #num_epochs = 2000
    #learning_rate = 1e-3
    num_epochs = 8000
    learning_rate = 1e-4
    
    # logistic regression model
    model = LogisticRegression(dim_x)
    
    loss_fn = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dtype_float = torch.FloatTensor
    
    X_train_torch = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
    Y_train_torch = Variable(torch.from_numpy(Y_train), requires_grad=False).type(dtype_float)
    
    X_valid_torch = Variable(torch.from_numpy(X_valid), requires_grad=False).type(dtype_float)
    #Y_valid_torch = Variable(torch.from_numpy(Y_valid), requires_grad=False).type(dtype_float)
        
    X_test_torch = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)
    #Y_test_torch = Variable(torch.from_numpy(Y_test), requires_grad=False).type(dtype_float)
    
    lambdas = np.arange(0.00001, 0.0003, 0.00001)
    #lambdas = np.arange(0.000001, 0.00003, 0.000001)
    
    best_train_acc = 0.0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_lambda = 0.0
    
    for lam in lambdas:
        print("Running model with lambda: " + str(lam))
        lam = np.array([lam])
        lam = Variable(torch.from_numpy(lam), requires_grad=True).type(dtype_float)
        for epoch in range(num_epochs):
            # Forward pass: Compute predicted y by passing x to the model
            Y_train_pred_torch = model(X_train_torch)
            
            # Compute L2 regularization term
            
            l2_reg = Variable(torch.FloatTensor([0]), requires_grad=True)
            for W in model.parameters():
                l2_reg = l2_reg + W.norm(2)**2
            
            
            # Compute and print loss
            loss = loss_fn(Y_train_pred_torch, Y_train_torch) + lam*l2_reg
            #print(epoch, loss.data[0])

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        Y_pred = Y_train_pred_torch.data.numpy()
        train_acc = accuracy_log(Y_pred, Y_train)
            
        Y_valid_pred_torch = model(X_valid_torch)
        Y_valid_pred = Y_valid_pred_torch.data.numpy()
        valid_acc = accuracy_log(Y_valid_pred, Y_valid)
            
        Y_test_pred_torch = model(X_test_torch)
        Y_test_pred = Y_test_pred_torch.data.numpy()
        test_acc = accuracy_log(Y_test_pred, Y_test)
            
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_lambda = lam.data.numpy()
            print("Current Best lambda: "+ str(best_lambda[0]))
            print("Current Best Training Accuracy: " + str(best_train_acc*100)+"%")
            print("Current Best Validation Accuracy: " + str(best_valid_acc*100)+"%")
            print("Current Best Test Accuracy: " + str(best_test_acc*100)+"%")
    
    
    print("Best lambda: "+ str(best_lambda[0]))
    print("Best Training Accuracy: " + str(best_train_acc*100)+"%")
    print("Best Validation Accuracy: " + str(best_valid_acc*100)+"%")
    print("Best Test Accuracy: " + str(best_test_acc*100)+"%")

def part4():
    print("Running Part 4 ...\n")
    '''
    # Logistic Regression Without Regularization
    torch.manual_seed(RAND_SEED)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
    train_keywords = all_words(real_train, fake_train)
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup_log(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
    dim_x = X_train.shape[1]
    num_epochs = 2500
    learning_rate = 1e-3
    
    # logistic regression model
    model = LogisticRegression(dim_x)
    
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dtype_float = torch.FloatTensor
    
    X_train_torch = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
    Y_train_torch = Variable(torch.from_numpy(Y_train), requires_grad=False).type(dtype_float)
    
    X_valid_torch = Variable(torch.from_numpy(X_valid), requires_grad=False).type(dtype_float)
    Y_valid_torch = Variable(torch.from_numpy(Y_valid), requires_grad=False).type(dtype_float)
        
    X_test_torch = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)
    Y_test_torch = Variable(torch.from_numpy(Y_test), requires_grad=False).type(dtype_float)
    
    train_perf_record = []
    valid_perf_record = []
    test_perf_record = []
    train_loss_record = []
    valid_loss_record = []
    test_loss_record = []
    iterations = []
    
    print("Running Logistic Regression Model Without Regularization")
    for epoch in range(num_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        Y_train_pred_torch = model(X_train_torch)
        
        # Compute and print loss
        loss = criterion(Y_train_pred_torch, Y_train_torch)
        #print(epoch, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            iterations.append(epoch)
            Y_pred = Y_train_pred_torch.data.numpy()
            train_acc = accuracy_log(Y_pred, Y_train)
            train_perf_record.append(train_acc)
            train_loss_record.append(loss.data[0])
            #print("Current Epoch: "+ str(epoch))
            #print("Current Training Accuracy: "+str(train_acc))
            #print("Current Training Loss: "+str(loss.data[0]))
            Y_valid_pred_torch = model(X_valid_torch)
            valid_loss = criterion(Y_valid_pred_torch, Y_valid_torch)
            Y_valid_pred = Y_valid_pred_torch.data.numpy()
            valid_acc = accuracy_log(Y_valid_pred, Y_valid)
            valid_perf_record.append(valid_acc)
            valid_loss_record.append(valid_loss.data[0])
            #print("Current Validation Accuracy: "+str(valid_acc))
            #print("Current Validation Loss: "+str(valid_loss.data[0]))
            Y_test_pred_torch = model(X_test_torch)
            test_loss = criterion(Y_test_pred_torch, Y_test_torch)
            Y_test_pred = Y_test_pred_torch.data.numpy()
            test_acc = accuracy_log(Y_test_pred, Y_test)
            test_perf_record.append(test_acc)
            test_loss_record.append(test_loss.data[0])
    
    print("Final Training Set Performance: "+str(train_perf_record[-1]*100)+"%")
    print("Final Validation Set Performance: "+str(valid_perf_record[-1]*100)+"%")
    print("Final Test Set Performance: "+str(test_perf_record[-1]*100)+"%")
    
    # Plotiing Performance
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(iterations, train_perf_record, label="Training Performance")
    ax1.plot(iterations, valid_perf_record, 'r-', label="Validation Performance")
    ax1.plot(iterations, test_perf_record, 'g-', label="Test Performance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Classification Performance for Real News vs Fake News")
    plt.legend(loc=0)
    plt.savefig("Part4_noReg_Performance")        
    
    # Plotting Cost
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(iterations, train_loss_record, label="Training Cost")
    ax2.plot(iterations, valid_loss_record, 'r-', label="Validation Cost")
    ax2.plot(iterations, test_loss_record, 'g-', label="Test Cost")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Classification Cost for Real News vs Fake News")
    plt.legend(loc=0)
    plt.savefig("Part4_noReg_Cost")
    '''
    print("\n")
    print("Running Logistic Regression Model With Regularization")
    print("Using Lambda: 0.0001")
    
    # Logistic Regression With Regularization
    torch.manual_seed(RAND_SEED)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
    train_keywords = all_words(real_train, fake_train)
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup_log(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
    lam = 0.0001
    dim_x = X_train.shape[1]
    num_epochs = 8000
    learning_rate = 1e-4
    
    # logistic regression model
    model = LogisticRegression(dim_x)
    
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dtype_float = torch.FloatTensor
    
    X_train_torch = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
    Y_train_torch = Variable(torch.from_numpy(Y_train), requires_grad=False).type(dtype_float)
    
    X_valid_torch = Variable(torch.from_numpy(X_valid), requires_grad=False).type(dtype_float)
    Y_valid_torch = Variable(torch.from_numpy(Y_valid), requires_grad=False).type(dtype_float)
        
    X_test_torch = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)
    Y_test_torch = Variable(torch.from_numpy(Y_test), requires_grad=False).type(dtype_float)
    
    train_perf_record = []
    valid_perf_record = []
    test_perf_record = []
    train_loss_record = []
    valid_loss_record = []
    test_loss_record = []
    iterations = []
    
    lam = np.array([lam])
    lam = Variable(torch.from_numpy(lam), requires_grad=True).type(dtype_float)
    
    for epoch in range(num_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        Y_train_pred_torch = model(X_train_torch)
        
        l2_reg = Variable(torch.FloatTensor([0]), requires_grad=True)
        for W in model.parameters():
            l2_reg = l2_reg + W.norm(2)**2
        
        # Compute and print loss
        loss = criterion(Y_train_pred_torch, Y_train_torch) + lam*l2_reg
        #print(epoch, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            iterations.append(epoch)
            Y_pred = Y_train_pred_torch.data.numpy()
            train_acc = accuracy_log(Y_pred, Y_train)
            train_perf_record.append(train_acc)
            train_loss_record.append(loss.data[0])
            Y_valid_pred_torch = model(X_valid_torch)
            valid_loss = criterion(Y_valid_pred_torch, Y_valid_torch)
            Y_valid_pred = Y_valid_pred_torch.data.numpy()
            valid_acc = accuracy_log(Y_valid_pred, Y_valid)
            valid_perf_record.append(valid_acc)
            valid_loss_record.append(valid_loss.data[0])
            Y_test_pred_torch = model(X_test_torch)
            test_loss = criterion(Y_test_pred_torch, Y_test_torch)
            Y_test_pred = Y_test_pred_torch.data.numpy()
            test_acc = accuracy_log(Y_test_pred, Y_test)
            test_perf_record.append(test_acc)
            test_loss_record.append(test_loss.data[0])
    print("Final Training Set Performance: "+str(train_perf_record[-1]*100)+"%")
    print("Final Validation Set Performance: "+str(valid_perf_record[-1]*100)+"%")
    print("Final Test Set Performance: "+str(test_perf_record[-1]*100)+"%")
    
    # Plotiing Performance
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(iterations, train_perf_record, label="Training Performance")
    ax3.plot(iterations, valid_perf_record, 'r-', label="Validation Performance")
    ax3.plot(iterations, test_perf_record, 'g-', label="Test Performance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Classification Performance for Real News vs Fake News with Reg")
    plt.legend(loc=0)
    plt.savefig("Part4_Reg_Performance")        
    
    # Plotting Cost
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(iterations, train_loss_record, label="Training Cost")
    ax4.plot(iterations, valid_loss_record, 'r-', label="Validation Cost")
    ax4.plot(iterations, test_loss_record, 'g-', label="Test Cost")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Classification Cost for Real News vs Fake News with Reg")
    plt.legend(loc=0)
    plt.savefig("Part4_Reg_Cost")
    
    # Save train_keywords for part 6.
    #print("Number of keywords: " + str(len(train_keywords)))
    pd.DataFrame(train_keywords).to_csv(P4_WORDS_CSV, index=False)
    
    # Save weights for part 6.
    # model.parameters() returns two array. First array has 4793 numbers, the second has just 1.
    # Logistic regression weights should be first array 
    # But model.parameters not scriptable
    i = 0
    for weights in model.parameters():
        if i == 0:
            '''
            print("weights is: ")
            print(weights)
            #Parameter containing:
            #0.5198  0.3776  0.5690  ...  -0.3093 -0.3571 -0.3570
            #[torch.FloatTensor of size 1x4793]
            '''
            for weight in weights:
                #weight is array of all weights
                weight = weight.data.numpy()
                #print("weight is: " + str(weight))
                weights_array = np.vstack((weight))
        i += 1
    # Drop first index of weight, which is for bias.
    # After dropping bias weight, the index for weight will 
    # match index for word in train_keywords.
    weights_array = weights_array[1:].flatten()
    #print("Weights array: ")
    #print(weights_array)
    #print("Weights array shape: " +str(weights_array.shape))
    pd.DataFrame(weights_array).to_csv(P4_WEIGHTS_CSV, index=False)
    
    print("\n")
    print("End of Part 4 \n")
"""===================================PART4==================================="""

"""===================================PART6==================================="""
def part6a():
    print("Running Part 6(a) ...\n")
    keywords = np.array(pd.read_csv(P4_WORDS_CSV)).flatten()
    weights = np.array(pd.read_csv(P4_WEIGHTS_CSV)).flatten()
    
    keywords_weights = {}
    # Build Dictionary
    for i in range(len(keywords)):
        keywords_weights[keywords[i]] = weights[i]

    top10_keywords, last10_keywords = top_keywords_log(keywords_weights, 10)
    print("Without Removing Stop-Words")
    # Words that appear alot in real news, but less so in fake news
    print("The top 10 words with the biggest weights: ")
    for word_weight in top10_keywords:
        print("Word: "+str(word_weight[0])+", Weight: "+str(word_weight[1]))
    print("\n")
    # Words that appear alot in fake news, but less so in real news
    print("The top 10 words with the lowest weights: ")
    for word_weight in last10_keywords:
        print("Word: "+str(word_weight[0])+", Weight: "+str(word_weight[1]))
    print("\n")
    print("End of Part 6(a) \n")

def part6b():
    print("Running Part 6(b) ...\n")
    keywords = np.array(pd.read_csv(P4_WORDS_CSV)).flatten()
    weights = np.array(pd.read_csv(P4_WEIGHTS_CSV)).flatten()
    
    keywords_weights = {}
    # Build Dictionary
    for i in range(len(keywords)):
        keywords_weights[keywords[i]] = weights[i]
    
    # Remove Stopwords
    keywords_weights = remove_stopwords_log(keywords_weights)
    
    top10_keywords, last10_keywords = top_keywords_log(keywords_weights, 10)
    print("After Removing Stop-Words \n")
    # Words that appear alot in real news, but less so in fake news
    print("The top 10 words with the biggest weights: ")
    for word_weight in top10_keywords:
        print("Word: "+str(word_weight[0])+", Weight: "+str(word_weight[1]))
    print("\n")
    # Words that appear alot in fake news, but less so in real news
    print("The top 10 words with the lowest weights: ")
    for word_weight in last10_keywords:
        print("Word: "+str(word_weight[0])+", Weight: "+str(word_weight[1]))
    print("\n")
    print("End of Part 6(b) \n")
"""===================================PART6==================================="""

"""===================================PART7==================================="""
def part7a():
    print("Running Part 7(a) ...\n")
    # Don't Remove stop-words
    max_depths = np.arange(10, 210, 10)
    #max_depths = np.arange(20, 60, 1)
    #max_depths = np.arange(80, 110, 10)
    #max_depths = np.array([90]) # Best
    
    # If Remove stop-words
    #max_depths = np.arange(60, 110, 1)
    #max_depths = np.array([100]) # Best
    
    best_train_acc = 0.0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_depth = 0.0
    
    max_depth_record = []
    train_perf_record = []
    valid_perf_record = []
    
    
    for depth in max_depths:
        print("Running Model With Max-Depth: " + str(depth))
        classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=depth, random_state = 0)
    
        real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
        train_keywords = all_words(real_train, fake_train)
        
        # Remove stop-words
        #train_keywords = remove_stopwords_part7(train_keywords)
        
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
        classifier.fit(X_train, Y_train)
        
        Y_train_pred = classifier.predict(X_train)
        Y_valid_pred = classifier.predict(X_valid)
        Y_test_pred = classifier.predict(X_test)
        
        train_acc = accuracy(Y_train_pred, Y_train)
        valid_acc = accuracy(Y_valid_pred, Y_valid)
        test_acc = accuracy(Y_test_pred, Y_test)
        
        max_depth_record.append(depth)
        train_perf_record.append(train_acc)
        valid_perf_record.append(valid_acc)
        
        if valid_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_depth = depth
            # Save Best Decision Tree Model
            with open('best_model.pkl', 'wb') as fid:
                cPickle.dump(classifier, fid)
            print("Current Best Max-Depth: " +str(depth))
            print("Current Best Training Performance: " +str(train_acc*100)+"%")
            print("Current Best Validation Performance: " +str(valid_acc*100)+"%")
            print("Current Best Test Performance: " +str(test_acc*100)+"%")
    print("\n")
    print("Best Max-Depth: " +str(best_depth))
    print("Best Training Performance: " +str(best_train_acc*100)+"%")
    print("Best Validation Performance: " +str(best_valid_acc*100)+"%")
    print("Best Test Performance: " +str(best_test_acc*100)+"%")
    
    # Plotiing Performance
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(max_depth_record, train_perf_record, label="Training Performance")
    ax1.plot(max_depth_record, valid_perf_record, 'r-', label="Validation Performance")
    plt.xlabel("Max-Dept Level")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Classification Performance vs Max-Depth")
    plt.legend(loc=0)
    plt.savefig("Part7_Performance_vs_Depth")  
    
    
    print("\n")
    print("End of Part 7(a) \n")
    
def part7b():
    print("Running Part 7(b) ...\n")
    # load best decision tree model from part 7(a)
    with open('best_model.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup()
    train_keywords = all_words(real_train, fake_train)
    export_graphviz(classifier, out_file="part7b.dot", max_depth=2, feature_names=train_keywords)
    
    print("\n")
    print("End of Part 7(b) \n")
"""===================================PART7===================================""" 

if __name__ == "__main__":
    part1()
    part2()
    part3a()
    part3b()
    #part4_best_model()
    part4()
    part6a()
    part6b()
    part7a()
    part7b()