import theano
import theano.tensor as T
import io
import json
import numpy as np
import os
import petname
import logging


def get_activation_function(activation):

    activations = ['tanh', 'sigmoid', 'relu', 'linear']

    if activation == activations[0]:
        return T.tanh
    elif activation == activations[1]:
        return T.nnet.sigmoid
    elif activation == activations[2]:
        return lambda x: x * (x > 0)
    elif activation == activations[3]:
        return lambda x: x
    else:
        raise NotImplementedError, \
        "Activation function not implemented. Choose one out of: {0}".format(activations)

def get_activation_function_derivative(activation):

    activations = ['tanh', 'sigmoid', 'relu', 'linear']

    if activation == activations[0]:
        return lambda x: 1-T.tanh(x)**2
    elif activation == activations[1]:
        return lambda x: T.nnet.sigmoid(x)*(1-T.nnet.sigmoid(x))
    elif activation == activations[2]:
        return lambda x: x > 0
    elif activation == activations[3]:
        return lambda x: T.ones(1)
    else:
        raise NotImplementedError, \
        "Activation function not implemented. Choose one out of: {0}".format(activations)



def parse_activations(activation_list):
    """From list of activation names for each layer return a list with the activation functions"""

    activation = [None]*len(activation_list)
    activation_prime = [None]*len(activation_list)
    for i, act in enumerate(activation_list):
        activation[i] = get_activation_function(act)
        activation_prime[i] = get_activation_function_derivative(act)

    return activation, activation_prime

def load_states(n):

    x = np.genfromtxt("data/states_1_len_61.txt", delimiter=',', dtype=theano.config.floatX)
    for i in xrange(2, n+1):
        tmp = np.genfromtxt("data/states_{0}_len_61.txt".format(i), delimiter=',', dtype=theano.config.floatX)
        x = np.vstack((x, tmp))
    return x

def load_controls(n):

    x = np.genfromtxt("data/controls_1_len_61.txt", delimiter=',', dtype=theano.config.floatX)
    for i in xrange(2, n+1):
        tmp = np.genfromtxt("data/controls_{0}_len_61.txt".format(i), delimiter=',', dtype=theano.config.floatX)
        x = np.vstack((x, tmp))
    return x

def log_init(path, session_name=None):
    if session_name is None:
        session_name = petname.Name()
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists("{0}/logs".format(path)):
            os.makedirs("{0}/logs".format(path))

        while os.path.isfile('{0}/logs/{1}.log'.format(path,session_name)):
            session_name = petname.Name()

    logging.basicConfig(level=logging.INFO, filename="{0}/logs/{1}.log".format(path, session_name),
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S")
    log = logging.getLogger(session_name)
    return log, session_name

def flatten(alist):
    t = []
    for element in alist:
        if not isinstance(element, list):
            t.append(element)
        else:
            t.extend(flatten(element))
    return t


def get_log_likelihood(output, y, precision, timeseries):
    if not timeseries:
        exp_value = -0.5*T.sum((output - y.dimshuffle('x',0,1))**2, axis=2)*precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value), -1*T.min(exp_value)),
                                         T.min(exp_value), T.max(exp_value))
    
        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value), axis=0)) +
                                        max_exp_value)#-\
                #       self.y.shape[0]*(T.log(self.m)+self.y.shape[1]/2.*T.log(2*np.pi))

    else:
        exp_value = -0.5*T.sum((output - y.dimshuffle(0, 'x',1, 2))**2, axis=3)*precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value), -1*T.min(exp_value)),
                                             T.max(exp_value), T.min(exp_value))
            
           
        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value), axis=1)) +
                                        max_exp_value)
    return log_likelihood
