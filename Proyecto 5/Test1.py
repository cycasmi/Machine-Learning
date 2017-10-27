# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:09:50 2017

@author: lobo4
"""

import numpy
 
# Initialize not-so-randomness
 numpy.random.seed(1)
 
# Sigmoid function : R -> [0-1]
def sigmoid(x, deriv=False):
    return x*(1-x) if deriv else 1/(1+numpy.exp(-x))
 
# Inputs
X = numpy.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
# Outputs
Y = numpy.array([[0,1,1,0]]).T
 
# Synapses
syn0 = 2*numpy.random.random((3,4)) - 1
syn1 = 2*numpy.random.random((4,1)) - 1
 
for j in range(60000):
    l0 = X
    l1 = sigmoid(numpy.dot(l0, syn0))
    l2 = sigmoid(numpy.dot(l1, syn1))
 
    # l2 correction
    l2_error = Y - l2
    l2_delta = l2_error * sigmoid(l2, True)
    syn1 += l1.T.dot(l2_delta)
 
    # l1 correction
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += l0.T.dot(l1_delta)
 
    if j % 10000 == 0:
        print ('Error : %s' % numpy.mean(numpy.abs(l2_error)))
print (l2, syn0, syn1)