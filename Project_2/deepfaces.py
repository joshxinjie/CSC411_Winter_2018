# -*- coding: utf-8 -*-

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import os
from scipy.ndimage import filters
import urllib.request
import hashlib
import glob as gl
import scipy.misc as sm
from scipy.misc import imread, imresize

import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import torch.nn as nn

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor
RAND_SEED = 7

''' ########## AlexNet ########## '''
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            

    def __init__(self, num_classes=6):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        return x
''' ########## End of AlexNet ########## '''

''' ########## Get Data ########## '''
def get_data():
    act = list(set([a.split("\n")[0] for a in open("subset_actors.txt").readlines()]))
    actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
    actresses = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
    testfile = urllib.request.URLopener()
    
    if not os.path.isdir('alex_uncropped'):
        os.makedirs('alex_uncropped')

    # Create cropped directory
    if not os.path.isdir('alex_cropped'):
        os.makedirs('alex_cropped')
        
    for a in act:
        name = a.split()[1].lower().strip()
        i = 0
        if a.strip() in actors:
            location = "facescrub_actors.txt"
        if a.strip() in actresses:
            location = "facescrub_actresses.txt"
        for line in open(location):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                try:
                    x1 = int(line.split()[-2].split(',')[0])
                    y1 = int(line.split()[-2].split(',')[1])
                    x2 = int(line.split()[-2].split(',')[2])
                    y2 = int(line.split()[-2].split(',')[3])
                    timeout(testfile.retrieve, (line.split()[4], "alex_uncropped/"+filename), {}, 45)
                    if not os.path.isfile("alex_uncropped/"+filename):
                        continue
                    print(filename)
                    
                    #h = hashlib.sha256()
                    #h.update(open("uncropped/"+filename).read())
                    #print(h.hexdigest())
                    #print(line.split()[6])
                    
                    img_hash = sha256("alex_uncropped/"+filename)
                    #print(img_hash)
                    #print(line.split()[6])
                    if str(img_hash) == line.split()[6]:
                    #if str(h.hexdigest()) == line.split()[6]:
                        # Cropping images and converting them to greyscale
                        image = imread("alex_uncropped/" + filename)
                        # Resize images to 227 by 227
                        CROPPEDimage = imresize(image[y1:y2,x1:x2], (227,227))
                        if CROPPEDimage.shape == (227,227,3):
                            imsave("alex_cropped/" + filename, CROPPEDimage)
                            print("Saved:", filename)
                        else:
                            CROPPEDimage = CROPPEDimage[:,:,:3]
                            imsave("alex_cropped/" + filename, CROPPEDimage)
                            print("Saved:", filename)
                    else:
                        print("SHA256 does not match")
                    i += 1
                except Exception:
                    continue

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def sha256(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()
''' ########## End of Get Data ########## '''

''' ########## Generate Training and Test Data ########## '''
def preprocess_images(name, training_set_size, validation_set_size, test_set_size):
    '''
    Returns training set, validation set and test set of images for given actor/actress
    '''
    np.random.seed(RAND_SEED)
    training_set = []
    validation_set = []
    test_set = []
    
    lastname = name.split()[1].lower().strip()
    
    images = gl.glob('alex_cropped/' + lastname + '*')
    np.random.shuffle(images)
    
    num_images = len(images)
    
    for i in range(num_images):
        im = imread(images[i])[:,:,:3] # Read an image
        im = im - np.mean(im.flatten()) # Flatten image into a vector
        im = im/np.max(np.abs(im.flatten())) # Normalize image
        proc_image = np.rollaxis(im, -1).astype(float32) # Swap red and blue channels
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

def generate_data(people, training_set_size, validation_set_size, test_set_size):
    i = 0
    for act in people:
        if (act == "Peri Gilpin") and (training_set_size > 37):
            training_size = 37
            validation_size = validation_set_size
            test_size = test_set_size
        else:
            training_size = training_set_size
            validation_size = validation_set_size
            test_size = test_set_size
        train_one_hot = np.zeros((training_size, 6))
        train_one_hot[:,i] = 1
        valid_one_hot = np.zeros((validation_size, 6))
        valid_one_hot[:,i] = 1
        test_one_hot = np.zeros((test_size, 6))
        test_one_hot[:,i] = 1
        training_set, validation_set, test_set = preprocess_images(act, training_size, validation_size, test_size)
        if i == 0:
            X_train = np.vstack([training_set])
            X_valid = np.vstack([validation_set])
            X_test = np.vstack([test_set])
            Y_train = np.vstack([train_one_hot])
            Y_valid = np.vstack([valid_one_hot])
            Y_test = np.vstack([test_one_hot])
        else:
            X_train = np.vstack((X_train, training_set))
            X_valid = np.vstack((X_valid, validation_set))
            X_test = np.vstack((X_test, test_set))
            Y_train = np.vstack((Y_train, train_one_hot))
            Y_valid = np.vstack((Y_valid, valid_one_hot))
            Y_test = np.vstack((Y_test, test_one_hot))
        i+=1
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
''' ########## End of Generate Training and Test Data ########## '''

def part10():
    print("Running Part 10 ...")
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    people = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_set_size = 70
    validation_set_size = 10
    test_set_size = 10
    
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = generate_data(people, training_set_size, validation_set_size, test_set_size)
    
    model_alex = MyAlexNet()
    model_alex.eval()

    # convert to torch variables
    train_x = Variable(torch.from_numpy(X_train).unsqueeze_(0), requires_grad=False)
    valid_x = Variable(torch.from_numpy(X_valid).unsqueeze_(0), requires_grad=False)
    test_x = Variable(torch.from_numpy(X_test).unsqueeze_(0), requires_grad=False)
    # remove original input to free memory
    del X_train
    del X_valid
    del X_test
    # return the activations from conv4
    y_conv4 = model_alex.forward(train_x[0]).data.numpy()
    valid_conv4 = model_alex.forward(valid_x[0]).data.numpy()
    test_conv4 = model_alex.forward(test_x[0]).data.numpy()

    x_train_conv4 = []
    x_valid_conv4 = []
    x_test_conv4 = []
    # flatten out the output from alex net and add 1 at the end
    for i in range(len(y_conv4)):
        x_train_conv4.append(np.insert(y_conv4[i].flatten(), 0, 1))
    x_train_conv4 = np.array(x_train_conv4)
    for i in range(len(valid_conv4)):
        x_valid_conv4.append(np.insert(valid_conv4[i].flatten(), 0, 1))
    x_valid_conv4 = np.array(x_valid_conv4)
    for i in range(len(test_conv4)):
        x_test_conv4.append(np.insert(test_conv4[i].flatten(), 0, 1))
    x_test_conv4 = np.array(x_test_conv4)

    # Optimal Network from part 8
    dim_x = 9217  # 256 * 6 * 6 + 1
    dim_h = 64 # Best
    #dim_h = 128
    #dim_h = 512
    #dim_h = 1024
    #dim_h = 4096
    dim_out = 6
    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h), \
                                torch.nn.ReLU(), \
                                torch.nn.Linear(dim_h, dim_out))
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    num_iter = 2000
    mini_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_perf_record = []
    valid_perf_record = []
    test_perf_record = []
    iterations = []
    print("Running Optimal NN Model Using")
    print("Activation Function: ReLU")
    print("Number of Hidden Units: " + str(dim_h))
    print("Learning Rate of: " + str(learning_rate))
    print("Mini-Batch Size: " + str(mini_size))
    for i in range(num_iter):
        train_idx = np.random.permutation(range(x_train_conv4.shape[0]))[:mini_size]
        X_mini_train = Variable(torch.from_numpy(x_train_conv4[train_idx]), requires_grad=False).type(dtype_float)
        Y_mini_train = Variable(torch.from_numpy(np.argmax(Y_train[train_idx], 1)), requires_grad=False).type(
            dtype_long)
        # Forward Pass
        Y_pred = model(X_mini_train)
        loss = loss_fn(Y_pred, Y_mini_train)
        # Zero out the previous gradient computation before running backpropagation
        model.zero_grad()
        # Backpropagration
        loss.backward()  # Compute the gradient
        optimizer.step()  # Use the gradient information to make a step
        if i % 100 == 0:
            train_perf = np.mean(np.argmax(Y_pred.data.numpy(), 1) == np.argmax(Y_train[train_idx], 1))
            train_perf_record.append(train_perf)
            # Convert preditons for validation set to numpy variables
            valid_Y_pred = model(Variable(torch.from_numpy(x_valid_conv4), requires_grad=False).type(dtype_float)).data.numpy()
            valid_perf = np.mean(np.argmax(valid_Y_pred, 1) == np.argmax(Y_valid, 1))
            valid_perf_record.append(valid_perf)
            # Convert preditons for test set to numpy variables
            test_Y_pred = model(Variable(torch.from_numpy(x_test_conv4), requires_grad=False).type(dtype_float)).data.numpy()
            test_perf = np.mean(np.argmax(test_Y_pred, 1) == np.argmax(Y_test, 1))
            test_perf_record.append(test_perf)
            iterations.append(i)

    print("Final Training Set Performance: " + str(train_perf_record[-1]))
    print("Final Validation Set Performance: " + str(valid_perf_record[-1]))
    print("Final Test Set Performance: " + str(test_perf_record[-1]))

    # Plotiing performance on training, validation and test sets for facial recognition
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(iterations, train_perf_record, label="Training Performance")
    ax1.plot(iterations, valid_perf_record, 'r-', label="Validation Performance")
    ax1.plot(iterations, test_perf_record, 'g-', label="Test Performance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Performance (% Accurate)")
    plt.title("Training, Validation and Test Performance for Facial Classification")
    plt.legend(loc=0)
    plt.savefig("Part10_Performance")
    print("End of Part 10")
    
if __name__ == "__main__":
    get_data()
    part10()
    