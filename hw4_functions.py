######
######  This file includes different functions used in HW3
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        value = 0;
        for i in range(features.shape[0]):
            value += max(1-labels[i]*(features[i]*w),0)
        value = value/features.shape[0]
        return value
    elif order==1:
        # value = ( TODO: value )
        value = math.inf
        #print(features.shape[1])
        subgradient = np.zeros((1,features.shape[1]))
        #print(subgradient.shape)
        for i in range(features.shape[0]):
            if (labels[i]*(features[i]*w) < 1):
                subgradient -= labels[i]*features[i]
        subgradient = subgradient/features.shape[0]
        subgradient = subgradient.T
        return (value,subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        value = 0;
        for i in range(features.shape[0]):
            value += max(1-labels[i]*(features[i]*w),0)
        value = value/features.shape[0]
        print("Value: ")
        print(value)
        return value
    elif order==1:
        # value = ( TODO: value )
        # subgradient = ( TODO: sungradient )
        value = math.inf
        #print(features.shape[1])
        subgradient = np.zeros((1,features.shape[1]))
        indices = np.random.choice(features.shape[0], minibatch_size, replace=False)
        for i in range(indices.shape[0]):
            if (labels[indices[i]]*(features[indices[i]]*w) < 1):
                subgradient -= labels[indices[i]]*features[indices[i]]
        subgradient = subgradient/minibatch_size
        subgradient = subgradient.T
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
