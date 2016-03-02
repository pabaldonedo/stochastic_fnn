import theano.tensor as T


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


def get_activation_derivative(activation):

    activations = ['tanh', 'sigmoid', 'relu', 'linear']

    if activation == activations[0]:
        return lambda x: 1-T.tanh(x)**2
    elif activation == activations[1]:
        return lambda x: T.nnet.sigmoid(x)*(1-T.nnet.sigmoid(x))
    elif activation == activations[2]:
        return lambda x: (x > 0)
    elif activation == activations[3]:
        return lambda x: 1
    else:
        raise NotImplementedError, \
        "Activation function not implemented. Choose one out of: {0}".format(activations)


def parse_activations(activations):
    sigma = [None]*len(activations)
    sigma_prime = [None]*len(activations)

    for i, a in enumerate(activations):
        sigma[i] = get_activation_function(a)
        sigma_prime[i] = get_activation_derivative(a)
    return sigma, sigma_prime
