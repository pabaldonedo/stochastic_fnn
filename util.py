import theano
import theano.tensor as T
import io
import json

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
        return lambda x: 1
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
