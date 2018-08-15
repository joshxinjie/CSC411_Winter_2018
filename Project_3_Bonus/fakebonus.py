# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:30:15 2018

"""

import numpy as np
import operator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from nltk import bigrams, trigrams
import matplotlib.pyplot as plt

RAND_SEED = 7

CLEAN_REAL = "clean_real.txt"
CLEAN_FAKE = "clean_fake.txt"
NEW_REAL = "new_real_2.txt"
NEW_FAKE = "new_fake_2.txt"
NB_WORDS_CSV = "NB_words.csv"
MLP_WORDS_CSV = "MLP_words.csv"
MLP_INPUTSIZE_CSV = "MLP_inputsize.csv"

"""###############################MLP#######################################"""
class MultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout, num_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                #nn.ReLU(),
                #nn.Dropout(dropout),
                #nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        out = self.classifier(x)
        out = F.sigmoid(out)
        return out


def setup_mlp():
    """
    Creates training, validation and test data in a 70/15/15 split.
    Only original data
    """
    np.random.seed(RAND_SEED)
    real_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_REAL).readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_FAKE).readlines()])
    real_rand = np.random.permutation(len(real_headlines))
    fake_rand = np.random.permutation(len(fake_headlines))
    real_train = real_headlines[real_rand[:int(round(0.7*len(real_rand)))]]
    real_valid = real_headlines[real_rand[int(round(0.7*len(real_rand))):int(round(0.85*len(real_rand)))]]
    real_test = real_headlines[real_rand[int(round(0.85*len(real_rand))):]]
    fake_train = fake_headlines[fake_rand[:int(round(0.7*len(fake_rand)))]]
    fake_valid = fake_headlines[fake_rand[int(round(0.7*len(fake_rand))):int(round(0.85*len(fake_rand)))]]
    fake_test = fake_headlines[fake_rand[int(round(0.85*len(fake_rand))):]]
    
    return real_train, real_valid, real_test, fake_train, fake_valid, fake_test    
 
def final_setup_mlp(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, all_keywords):
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
            #print(x_curr)
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

def word_dict(headlines, limit=None, bi=False, tri=False):
    """
    Returns dictionary of unigrams and bigrams
    If limit is not none, return top 'limit' of unigrams and bigrams
    
    Creating bigrams
    text = "to be or not to be"
    string_bigrams = bigrams(text.split())
    list(string_bigrams)
    Out[21]: [('to', 'be'), ('be', 'or'), ('or', 'not'), ('not', 'to'), ('to', 'be')]
    """
    unique_words = {}
    for headline in headlines:
        words = headline.split()
        for word in words:
            if word not in unique_words:
                unique_words[word] = 1
            else:
                unique_words[word] += 1
        if bi == True:
            headline_bigrams = bigrams(words)
            for bigrm in headline_bigrams:
                if bigrm not in unique_words:
                    unique_words[bigrm] = 1
                else:
                    unique_words[bigrm] += 1
        if tri == True:
            headline_trigrams = trigrams(words)
            for trigrm in headline_trigrams:
                if trigrm not in unique_words:
                    unique_words[trigrm] = 1
                else:
                    unique_words[trigrm] += 1
    words_list = []
    if limit==None:
        # No vocabulary size limit
        words_list = list(unique_words.keys())
    else:
        # If we have vocabulary size limit
        sorted_words = sorted(unique_words.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(limit):
            words_list.append(sorted_words[i][0])
    return words_list

def accuracy_mlp(Y_pred, Y_act):
    """
    Returns accuracy of predictions for Logistic Regression
    """
    accuracy = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] > 0.5 and Y_act[i] == 1:
            accuracy += 1
        elif Y_pred[i] < 0.5 and Y_act[i] == 0:
            accuracy += 1
    return accuracy/float(len(Y_pred))

def mlp_grid():
    """
    Grid Search for best MLP model
    """
    torch.manual_seed(RAND_SEED)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup_mlp()
    train = np.concatenate((real_train, fake_train))
    '''Set vocabulary size limit here'''
    bigrm = True
    trigrm = True
    train_keywords = word_dict(train,limit=None,bi=bigrm,tri=trigrm)
    #print(train_keywords)
    print(len(train_keywords))
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup_mlp(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
    input_size = X_train.shape[1]
    hidden_units = [64, 128, 512, 1024]
    dropouts = np.arange(0.1, 0.6, 0.1)
    num_classes = 1
    num_epochs = np.arange(10, 35, 5)
    #num_epochs = np.arange(30, 35, 5)

    learning_rate = 1e-3
    
    best_train_acc = 0
    best_valid_acc = 0
    best_test_acc = 0
    best_hidden_size = 0
    best_dropout = 0
    best_epochs = 0
    
    for epochs in num_epochs:
        for hidden_size in hidden_units:
            for dropout in dropouts:
                print("Running Model with "+str(hidden_size)+" hidden units and "+str(dropout)+" dropout prob and "+str(epochs)+" epochs")
                model = MultiLayerPerceptron(input_size, hidden_size, dropout, num_classes)
                
                loss_fn = torch.nn.BCELoss(size_average=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                dtype_float = torch.FloatTensor
                
                X_train_torch = Variable(torch.from_numpy(X_train), requires_grad=False).type(dtype_float)
                Y_train_torch = Variable(torch.from_numpy(Y_train), requires_grad=False).type(dtype_float)
                
                X_valid_torch = Variable(torch.from_numpy(X_valid), requires_grad=False).type(dtype_float)
                #Y_valid_torch = Variable(torch.from_numpy(Y_valid), requires_grad=False).type(dtype_float)
                    
                X_test_torch = Variable(torch.from_numpy(X_test), requires_grad=False).type(dtype_float)
                #Y_test_torch = Variable(torch.from_numpy(Y_test), requires_grad=False).type(dtype_float)
                
                for epoch in range(epochs):
                    # Forward pass: Compute predicted y by passing x to the model
                    Y_train_pred_torch = model(X_train_torch)      
                    
                    # Compute and print loss
                    loss = loss_fn(Y_train_pred_torch, Y_train_torch)
                    #print(epoch, loss.data[0])
            
                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                Y_pred = Y_train_pred_torch.data.numpy()
                train_acc = accuracy_mlp(Y_pred, Y_train)
                Y_valid_pred_torch = model(X_valid_torch)
                Y_valid_pred = Y_valid_pred_torch.data.numpy()
                valid_acc = accuracy_mlp(Y_valid_pred, Y_valid)
                Y_test_pred_torch = model(X_test_torch)
                Y_test_pred = Y_test_pred_torch.data.numpy()
                test_acc = accuracy_mlp(Y_test_pred, Y_test)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_train_acc = train_acc
                    best_test_acc = test_acc
                    best_hidden_size = hidden_size
                    best_dropout = dropout
                    best_epochs = epochs
                    print("Current Best Epochs: " + str(epochs))
                    print("Current Best Hidden Units: " + str(hidden_size))
                    print("Current Best Dropout: "+str(dropout))
                    print("Current Best Training Accuracy: " + str(best_train_acc*100)+"%")
                    print("Current Best Validation Accuracy: " + str(best_valid_acc*100)+"%")
                    print("Current Best Test Accuracy: " + str(best_test_acc*100)+"%")
    print("Best Epochs: " + str(best_epochs))
    print("Best Hidden Units: " + str(best_hidden_size))
    print("Best Dropout: "+str(best_dropout))
    print("Best Training Accuracy: " + str(best_train_acc*100)+"%")
    print("Best Validation Accuracy: " + str(best_valid_acc*100)+"%")
    print("Best Test Accuracy: " + str(best_test_acc*100)+"%")


def mlp_model():
    """
    Best MLP Model
    """
    print("Preprocessing Data")
    torch.manual_seed(RAND_SEED)
    real_train, real_valid, real_test, fake_train, fake_valid, fake_test = setup_mlp()
    train = np.concatenate((real_train, fake_train))
    '''Set vocabulary size limit here'''
    bigrm = True
    trigrm = True
    train_keywords = word_dict(train,limit=None,bi=bigrm,tri=trigrm)
    print("vocabulary size: "+str(len(train_keywords)))
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = final_setup_mlp(real_train, real_valid, real_test, fake_train, fake_valid, fake_test, train_keywords)
    
    input_size = X_train.shape[1]
    print("Input size: "+str(input_size))
    hidden_size = 1024
    num_classes = 1
    num_epochs = 10
    dropout = 0.4
    learning_rate = 1e-3
    
    model = MultiLayerPerceptron(input_size, hidden_size, dropout, num_classes)
    
    loss_fn = torch.nn.BCELoss(size_average=True)
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
    
    print("Data Processed")
    print("Running Model")
    
    for epoch in range(num_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        Y_train_pred_torch = model(X_train_torch)      
        
        # Compute and print loss
        loss = loss_fn(Y_train_pred_torch, Y_train_torch)
        #print(epoch, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            iterations.append(epoch)
            Y_pred = Y_train_pred_torch.data.numpy()
            train_acc = accuracy_mlp(Y_pred, Y_train)
            train_perf_record.append(train_acc)
            train_loss_record.append(loss.data[0])
            Y_valid_pred_torch = model(X_valid_torch)
            valid_loss = loss_fn(Y_valid_pred_torch, Y_valid_torch)
            Y_valid_pred = Y_valid_pred_torch.data.numpy()
            valid_acc = accuracy_mlp(Y_valid_pred, Y_valid)
            valid_perf_record.append(valid_acc)
            valid_loss_record.append(valid_loss.data[0])
            Y_test_pred_torch = model(X_test_torch)
            test_loss = loss_fn(Y_test_pred_torch, Y_test_torch)
            Y_test_pred = Y_test_pred_torch.data.numpy()
            test_acc = accuracy_mlp(Y_test_pred, Y_test)
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
    plt.title("Classification Performance for Real News vs Fake News")
    plt.legend(loc=0)
    plt.savefig("MLP_Performance")        
    
    # Plotting Cost
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(iterations, train_loss_record, label="Training Cost")
    ax4.plot(iterations, valid_loss_record, 'r-', label="Validation Cost")
    ax4.plot(iterations, test_loss_record, 'g-', label="Test Cost")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Classification Cost for Real News vs Fake News")
    plt.legend(loc=0)
    plt.savefig("MLPCost")
    
"""###############################MLP#######################################"""

"""###############################SVM#######################################"""
def initial_setup():
    """
    Creates training, validation and test data in a 70/15/15 split.
    """
    np.random.seed(RAND_SEED)
    real_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_REAL).readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_FAKE).readlines()])
    headlines = np.concatenate((real_headlines, fake_headlines))
    
    labels = np.zeros(len(real_headlines)+len(fake_headlines))
    labels[:len(real_headlines)] = 1
    labels = labels.flatten()
    
    return headlines, labels

def final_setup(headlines, labels, gram, mx_df, mn_df):
    
    #tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=gram,max_df=mx_df,min_df=mn_df)
    tfidf = TfidfVectorizer(ngram_range=gram,max_df=mx_df,min_df=mn_df)
    
    headlines_tfidf = tfidf.fit_transform(headlines)
    
    X_train_tfidf, X_valid_test_tfidf, Y_train, Y_valid_test = train_test_split(headlines_tfidf, labels, test_size = 0.3, random_state=1234)
    X_valid_tfidf, X_test_tfidf, Y_valid, Y_test = train_test_split(X_valid_test_tfidf, Y_valid_test, test_size = 0.5, random_state=1234)

    return X_train_tfidf, X_valid_tfidf, X_test_tfidf, Y_train, Y_valid, Y_test, tfidf

def initial_setup_svm_part2():
    """
    Creates training, validation and test data in a 70/15/15 split. Includes new dataset
    """
    np.random.seed(RAND_SEED)
    real_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_REAL).readlines()])
    fake_headlines = np.array([a.split("\n")[0] for a in open(CLEAN_FAKE).readlines()])
    new_real_headlines = np.array([a.split("\n")[0] for a in open(NEW_REAL).readlines()])
    new_fake_headlines = np.array([a.split("\n")[0] for a in open(NEW_FAKE).readlines()])
    real_headlines = np.concatenate((real_headlines, new_real_headlines))
    fake_headlines = np.concatenate((fake_headlines, new_fake_headlines))
    headlines = np.concatenate((real_headlines, fake_headlines))
    
    labels = np.zeros(len(real_headlines)+len(fake_headlines))
    labels[:len(real_headlines)] = 1
    labels = labels.flatten()
    
    return headlines, labels

def final_setup_new_svm(headlines, labels, tfidf):
    headlines_tfidf = tfidf.fit_transform(headlines)
    
    X_train_tfidf, X_valid_test_tfidf, Y_train, Y_valid_test = train_test_split(headlines_tfidf, labels, test_size = 0.3, random_state=RAND_SEED)
    X_valid_tfidf, X_test_tfidf, Y_valid, Y_test = train_test_split(X_valid_test_tfidf, Y_valid_test, test_size = 0.5, random_state=RAND_SEED)

    return X_train_tfidf, X_valid_tfidf, X_test_tfidf, Y_train, Y_valid, Y_test

def accuracy_svm(Y_pred, Y_act):
    """
    Returns Accuracy of Predictions For SVM, xgboost and Random Forest
    """
    accuracy = 0.0
    for i in range(len(Y_pred)):
        accuracy += (Y_pred[i] == Y_act[i])
    return (accuracy/float(len(Y_pred)))

def svm_grid():
    """
    Grid Search for Best SVM
    """
    
    ngram_range = [(1,1),(1,2),(1,3)]
    max_df = [0.7,0.8,0.9,0.99]
    min_df = [0.001,0.01,0.1]
    #max_df = [0.999]
    #min_df = [0.001]
    cs = np.arange(0.1, 1.5, 0.1)
    gammas = np.arange(0.1, 1.1, 0.1)
    
    best_train_acc = 0.0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_gram = 0.0
    best_mx_df = 0.0
    best_mn_df = 0.0
    best_c = 0.0
    best_gamma = 0.0
    
    headlines, labels = initial_setup()
    
    for c in cs:
        for gamma in gammas:
            for gram in ngram_range:
                for mx_df in max_df:
                    for mn_df in min_df:
                            print("Running SVM Model With")
                            print("c: " +str(c)+ ", gamma: "+str(gamma)+", gram: "+str(gram)+", mx_df: "+str(mx_df)+", mn_df: "+str(mn_df))
                            X_train, X_valid, X_test, Y_train, Y_valid, Y_test, vocab = final_setup(headlines, labels, gram, mx_df, mn_df)
                            svc = svm.SVC(kernel='rbf', C=c,gamma=gamma)
                            
                            svc.fit(X_train, Y_train)
                                            
                            Y_train_pred = svc.predict(X_train)
                            Y_valid_pred = svc.predict(X_valid)
                            Y_test_pred = svc.predict(X_test)
                            
                            train_acc = accuracy_svm(Y_train_pred, Y_train)
                            valid_acc = accuracy_svm(Y_valid_pred, Y_valid)
                            test_acc = accuracy_svm(Y_test_pred, Y_test)
                            if valid_acc > best_valid_acc:
                                best_train_acc = train_acc
                                best_valid_acc = valid_acc
                                best_test_acc = test_acc
                                best_c = c
                                best_gamma = gamma
                                best_gram = gram
                                best_mx_df = mx_df
                                best_mn_df = mn_df
                                print("Current Best c: " + str(c))
                                print("Current Best gamma: " + str(gamma))
                                print("Current Best gram: " + str(gram))
                                print("Current Best mx_df: " +str(mx_df))
                                print("Current Best mn_df: " +str(mn_df))
                                print("Current Best Training Performance: " +str(train_acc))
                                print("Current Best Validation Performance: " +str(valid_acc))
                                print("Current Best Test Performance: " +str(test_acc))
    print("Best c: " + str(best_c))
    print("Best gamma: " + str(best_gamma))
    print("Best gram: " + str(best_gram))
    print("Best mx_df: " +str(best_mx_df))
    print("Best mn_df: " +str(best_mn_df))
    print("Best Training Performance: " +str(best_train_acc))
    print("Best Validation Performance: " +str(best_valid_acc))
    print("Best Test Performance: " +str(best_test_acc))
    
def svm_model():
    """
    Best Model
    """
    
    c = 0.9
    gamma = 0.5
    gram = (1,1)
    mx_df = 0.99
    mn_df = 0.001
    
    print("Running SVM Model With")
    print("c: " +str(c)+ ", gamma: "+str(gamma)+", gram: "+str(gram)+", mx_df: "+str(mx_df)+", mn_df: "+str(mn_df))
    #headlines, labels = initial_setup_svm_part2()
    headlines, labels = initial_setup()
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, tfidf = final_setup(headlines, labels, gram, mx_df, mn_df)
    svc = svm.SVC(kernel='rbf', C=c,gamma=gamma)
    
    Y_train = Y_train.flatten()
    Y_valid = Y_valid.flatten()
    Y_test = Y_test.flatten()
    
    svc.fit(X_train, Y_train)
                    
    Y_train_pred = svc.predict(X_train)
    Y_valid_pred = svc.predict(X_valid)
    Y_test_pred = svc.predict(X_test)
    
    train_acc = accuracy_svm(Y_train_pred, Y_train)
    valid_acc = accuracy_svm(Y_valid_pred, Y_valid)
    test_acc = accuracy_svm(Y_test_pred, Y_test)

    print("Training Performance: " +str(train_acc))
    print("Validation Performance: " +str(valid_acc))
    print("Test Performance: " +str(test_acc))
        
"""###############################SVM#######################################"""

"""###############################RF#######################################"""
def randomforest_grid():
    """
    Best gram: (1, 1)
    Best mx_df: 0.99
    Best mn_df: 0.001
    Best Num-Tree: 20
    Best Training Performance: 0.986858838491
    Best Validation Performance: 0.812949640288
    Best Test Performance: 0.769387755102
    """
    num_trees = np.arange(10, 110, 10)
    #max_depths = np.arange(20, 60, 1)
    #max_depths = np.arange(80, 110, 10)
    #max_depths = np.array([90]) # Best
    
    ngram_range = [(1,1),(1,2),(1,3)]
    max_df = [0.7,0.8,0.9,0.99]
    min_df = [0.001,0.01,0.1]

    
    best_train_acc = 0.0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_gram = 0.0
    best_mx_df = 0.0
    best_mn_df = 0.0
    best_num_tree = 0.0
    
    for gram in ngram_range:
        for mx_df in max_df:
            for mn_df in min_df:
                for num_tree in num_trees:
                    print("Running Random Forest Model With")
                    print("gram: "+str(gram)+", mx_df: "+str(mx_df)+", mn_df: "+str(mn_df)+", num trees: "+ str(num_tree))
                    classifier = RandomForestClassifier(n_estimators = num_tree, criterion = 'entropy', random_state = RAND_SEED)
                    
                    headlines, labels = initial_setup()
                    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, vocab = final_setup(headlines, labels, gram, mx_df, mn_df)
                
                    classifier.fit(X_train, Y_train)
                    
                    Y_train_pred = classifier.predict(X_train)
                    Y_valid_pred = classifier.predict(X_valid)
                    Y_test_pred = classifier.predict(X_test)
                    
                    train_acc = accuracy_svm(Y_train_pred, Y_train)
                    valid_acc = accuracy_svm(Y_valid_pred, Y_valid)
                    test_acc = accuracy_svm(Y_test_pred, Y_test)
                    
                    if valid_acc > best_valid_acc:
                        best_train_acc = train_acc
                        best_valid_acc = valid_acc
                        best_test_acc = test_acc
                        best_gram = gram
                        best_mx_df = mx_df
                        best_mn_df = mn_df
                        best_num_tree = num_tree
                        print("Current Best gram: " + str(gram))
                        print("Current Best mx_df: " +str(mx_df))
                        print("Current Best mn_df: " +str(mn_df))
                        print("Current Best Num-Tree: " +str(num_tree))
                        print("Current Best Training Performance: " +str(train_acc))
                        print("Current Best Validation Performance: " +str(valid_acc))
                        print("Current Best Test Performance: " +str(test_acc))
    print("Best gram: " + str(best_gram))
    print("Best mx_df: " +str(best_mx_df))
    print("Best mn_df: " +str(best_mn_df))
    print("Best Num-Tree: " +str(best_num_tree))
    print("Best Training Performance: " +str(best_train_acc))
    print("Best Validation Performance: " +str(best_valid_acc))
    print("Best Test Performance: " +str(best_test_acc))
"""###############################RF#######################################"""

"""###############################xgboost#####################################"""
def xgboost_grid():
    """
    Best gram: (1, 1)
    Best mx_df: 0.99
    Best mn_df: 0.001
    Best Training Performance: 0.802458668928
    Best Validation Performance: 0.772182254197
    Best Test Performance: 0.744897959184
    """
    ngram_range = [(1,1),(1,2),(1,3)]
    max_df = [0.7,0.8,0.9,0.99]
    min_df = [0.001,0.01,0.1]
    
    best_train_acc = 0.0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_gram = 0.0
    best_mx_df = 0.0
    best_mn_df = 0.0
    
    for gram in ngram_range:
        for mx_df in max_df:
            for mn_df in min_df:
                print("Running XGBOOST Model With")
                print("gram: "+str(gram)+", mx_df: "+str(mx_df)+", mn_df: "+str(mn_df))
                headlines, labels = initial_setup()
                X_train, X_valid, X_test, Y_train, Y_valid, Y_test, vocab = final_setup(headlines, labels, gram, mx_df, mn_df)
                xgb_classifier = XGBClassifier()
                
                xgb_classifier.fit(X_train, Y_train)
                                
                Y_train_pred = xgb_classifier.predict(X_train)
                Y_valid_pred = xgb_classifier.predict(X_valid)
                Y_test_pred = xgb_classifier.predict(X_test)
                
                train_acc = accuracy_svm(Y_train_pred, Y_train)
                valid_acc = accuracy_svm(Y_valid_pred, Y_valid)
                test_acc = accuracy_svm(Y_test_pred, Y_test)
                if valid_acc > best_valid_acc:
                        best_train_acc = train_acc
                        best_valid_acc = valid_acc
                        best_test_acc = test_acc
                        best_gram = gram
                        best_mx_df = mx_df
                        best_mn_df = mn_df
                        print("Current Best gram: " + str(gram))
                        print("Current Best mx_df: " +str(mx_df))
                        print("Current Best mn_df: " +str(mn_df))
                        print("Current Best Training Performance: " +str(train_acc))
                        print("Current Best Validation Performance: " +str(valid_acc))
                        print("Current Best Test Performance: " +str(test_acc))
    print("Best gram: " + str(best_gram))
    print("Best mx_df: " +str(best_mx_df))
    print("Best mn_df: " +str(best_mn_df))
    print("Best Training Performance: " +str(best_train_acc))
    print("Best Validation Performance: " +str(best_valid_acc))
    print("Best Test Performance: " +str(best_test_acc))
"""###############################xgboost#####################################"""

if __name__ == "__main__":
    #mlp_grid()
    mlp_model()
    #svm_grid()
    svm_model()
    #randomforest_grid()
    #xgboost_grid()