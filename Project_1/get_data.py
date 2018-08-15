
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
from rgb2gray import rgb2gray
import urllib.request
import hashlib


act = list(set([a.split("\n")[0] for a in open("subset_actors.txt").readlines()]))

actors = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
actresses =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']

''' Helper Functions '''
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


testfile = urllib.request.URLopener()            

#Note: you need to create the uncropped folder first in order 
#for this to work

# Create uncropped directory
if not os.path.isdir('uncropped'):
    os.makedirs('uncropped')

# Create cropped directory
if not os.path.isdir('cropped'):
    os.makedirs('cropped')

# Create cropped male directory
if not os.path.isdir('cropped/male'):
    os.makedirs('cropped/male')
    
# Create cropped female directory
if not os.path.isdir('cropped/female'):
    os.makedirs('cropped/female')

for a in act:
    name = a.split()[1].lower().strip()
    i = 0
    if a.strip() in actors:
        location = "facescrub_actors.txt"
    if a.strip() in actresses:
        location = "facescrub_actresses.txt"
    sha256_array = []
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
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                print(filename)
                
                img_hash = sha256("uncropped/"+filename)
                # Remove all previous duplicate images, since they tend to have bad bounding boxes
                while(img_hash in sha256_array):
                    j = sha256_array.index(img_hash)
                    prev_filename = name+str(j)+'.'+line.split()[4].split('.')[-1]
                    print("Previous duplicate file:", prev_filename)
                    if a in actors:
                        os.remove("cropped/male/"+prev_filename)
                        print("Deleted:", prev_filename)
                        sha256_array.remove(img_hash)
                    elif a in actresses:
                        os.remove("cropped/female/"+prev_filename)
                        print("Deleted:", prev_filename)
                        sha256_array.remove(img_hash)
                sha256_array.append(img_hash)
                
                # Cropping images and converting them to greyscale                
                image = imread("uncropped/" + filename)
                GSimage = rgb2gray(image)
                # Resize images to 32 by 32
                CROPPEDimage = imresize(GSimage[y1:y2,x1:x2], (32,32))
                if a in actors:
                    imsave("cropped/male/" + filename, CROPPEDimage)
                    print("Saved:", filename)
                else:
                    imsave("cropped/female/" + filename, CROPPEDimage)
                    print("Saved:", filename)
                i += 1
            except Exception:
                continue
    
            
    
    