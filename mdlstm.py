#!/usr/bin/python 

'''Usage of multidimensional lstm from pybrain. Basically a control experiment'''

import numpy as np
import matplotlib 
matplotlib.use('Agg')

from pylab import *
from matplotlib import pyplot as plt
from pybrain.structure.networks.multidimensional import *
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.classification import SupervisedDataSet

import cPickle 

print "Loading the dataset..."

f = open('../deblurring/sample_train.pkl', 'rb')
train_x, train_y = cPickle.load(f)
f.close()

f = open('../deblurring/sample_test.pkl', 'rb')
test_x, test_y = cPickle.load(f)
f.close()

print "Dataset loaded successfully..."

mdlstm = MultiDimensionalLSTM((28,28))
ds = SupervisedDataSet(train_x, train_y)

trainer = BackpropTrainer(mdlstm, ds, momentum = 0.1, verbose=True)
epochs  = 100

for epoch in range(epochs):
  trainer.trainEpochs(1)

print "Showing some sample outputs..."

img = test_x[10]
targ = test_y[10]

outp = mdlstm.activate(img)

fig = plt.figure()
plt.subplot(131)
plt.imshow(img.reshape((28,28)), cmap='gray')

plt.subplot(132)
plt.imshow(targ.reshape((28,28)), cmap='gray')

plt.subplot(133)
plt.imshow(outp.reshape((28,28)), cmap='gray')

fig.savefig('sample_output.jpg')
print "Figure saved successfully...")
