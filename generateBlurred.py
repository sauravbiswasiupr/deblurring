#!/usr/bin/python 
'''Script to generate random images with grayscale values in the range 0-1 and blur them with sigma which has been input by the user'''

import argparse
import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from scipy.ndimage.filters import gaussian_filter
import cPickle

parser = argparse.ArgumentParser()
parser.add_argument('sigma', type=float, help = 'the blur value to blur the image with')
parser.add_argument('trainsize', type=int, help = 'training set size')
parser.add_argument('testsize', type=int, help='test set size')
parser.add_argument('filename', type=str, help ='the filename where you want to store the pickled files')

args = parser.parse_args()
sigma = args.sigma
filename = args.filename
trainsize = args.trainsize
testsize = args.testsize

maininps = []
maintargs = []

print "Sigma used:", sigma

for i in range(trainsize+testsize):
  img = np.random.ranf((28,28))
  out = gaussian_filter(img, sigma)
  out = out.reshape((28*28,))
  img = img.reshape((28*28,))
  maininps.append(img)
  maintargs.append(out)

seq = [i for i in range(len(maininps))]
np.random.shuffle(seq)

train_x = []; train_y = []
test_x = []; test_y = []

print 
for i in seq[:trainsize]:
  train_x.append(maininps[i])
  train_y.append(maintargs[i])

for i in seq[trainsize:]:
  test_x.append(maininps[i])
  test_y.append(maintargs[i])

train_x,train_y,test_x,test_y = map(lambda x: np.array(x), [train_x, train_y, test_x, test_y])

train = (train_x,train_y)
test = (test_x, test_y)

print "Training and test sets created. Saving to disk"

trainfile = filename + '_train.pkl'
f = open(trainfile, 'wb')
cPickle.dump(train, f)
f.close()

testfile = filename + '_test.pkl'
f = open(testfile, 'wb')
cPickle.dump(test, f)
f.close()

print "Saved successfully to disk..."
