import theano
import theano.tensor as T
import theano.ifelse
import numpy as np
import os
import petname
import logging


def get_activation_function(activation):
    """Returns function or theano function that corresponds to activation.

    :type activation: string.
    :param activation: activation function name.
    """
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
        raise NotImplementedError(
            "Activation function not implemented. Choose one out of: {0}".
            format(activations))


def get_activation_function_derivative(activation):
    """Returns function or theano function that corresponds to the derivative of
    activation.

    :type activation: string.
    :param activation: activation function name.
    """
    activations = ['tanh', 'sigmoid', 'relu', 'linear']

    if activation == activations[0]:
        return lambda x: 1 - T.tanh(x)**2
    elif activation == activations[1]:
        return lambda x: T.nnet.sigmoid(x) * (1 - T.nnet.sigmoid(x))
    elif activation == activations[2]:
        return lambda x: x > 0
    elif activation == activations[3]:
        return lambda x: T.ones(1)
    else:
        raise NotImplementedError(
            "Activation function not implemented. Choose one out of: {0}".
            format(activations))


def parse_activations(activation_list):
    """From list of activation names for each layer return a list with the
    activation functions

    :type activation_list: list
    """

    activation = [None] * len(activation_list)
    activation_prime = [None] * len(activation_list)
    for i, act in enumerate(activation_list):
        activation[i] = get_activation_function(act)
        activation_prime[i] = get_activation_function_derivative(act)

    return activation, activation_prime


def load_states(n):
    """Loads into np.array all the files states_i_len_61 from 1 to n.

    :type n: int
    """

    x = np.genfromtxt("data/states_1_len_61.txt",
                      delimiter=',', dtype=theano.config.floatX)
    for i in xrange(2, n + 1):
        tmp = np.genfromtxt(
            "data/states_{0}_len_61.txt".format(i), delimiter=',',
            dtype=theano.config.floatX)
        x = np.vstack((x, tmp))
    return x


def load_controls(n):
    """Loads into np.array all the files controls_i_len_61 from 1 to n.

    :type n: int
    """

    x = np.genfromtxt("data/controls_1_len_61.txt",
                      delimiter=',', dtype=theano.config.floatX)
    for i in xrange(2, n + 1):
        tmp = np.genfromtxt(
            "data/controls_{0}_len_61.txt".format(i), delimiter=',',
            dtype=theano.config.floatX)
        x = np.vstack((x, tmp))
    return x


def load_files(n, fname):
    """Loads into np.array all the files controls_i_len_61 from 1 to n.

    :type n: int
    """

    x = np.genfromtxt("data/{0}_1_len_61.txt".format(fname),
                      delimiter=',', dtype=theano.config.floatX)
    for i in xrange(2, n + 1):
        tmp = np.genfromtxt(
            "data/{0}_{1}_len_61.txt".format(fname, i), delimiter=',',
            dtype=theano.config.floatX)
        x = np.vstack((x, tmp))
    return x


def log_init(path, session_name=None):
    """Initializes logging module.

    :type path: string.
    :param path: path where to write the log file.

    :type session_name: string or None.
    :param session_name: log_file name. If None one name is generated with
                                                                   petname.
    """

    if session_name is None:
        session_name = petname.Name()
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists("{0}/logs".format(path)):
            os.makedirs("{0}/logs".format(path))

        while os.path.isfile('{0}/logs/{1}.log'.format(path, session_name)):
            session_name = petname.Name()

    logging.basicConfig(level=logging.INFO, filename="{0}/logs/{1}.log".format(
                        path, session_name),
                        format="[%(levelname)s] %(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S")
    log = logging.getLogger(session_name)
    return log, session_name


def flatten(alist):
    """Flattens a list of lists of any depth to one single list.

    :type alist list.
    """
    t = []
    for element in alist:
        if not isinstance(element, list):
            t.append(element)
        else:
            t.extend(flatten(element))
    return t


def get_log_likelihood(output, y, precision, timeseries):
    """outputs theano variable that contains the UNNORMALIZED log-likelihood.

    :type output: theano.tensor3 if timeseries False/ theano.tensor4 if
                                                               timeseries True.
    :param output: theano output variable of network.

    :type y: theano.matrix.
    :param y: theano variable containing true output.

    :type precision: float.
    :param precision: precision parameter of gaussian distribution.
                      Useful to overcome numerical problems.
                      If getting nans try to increase its value.

    :type timeseries: bool.
    :param timeseries: tells if the network is recurrent (True) or feedforward
                                                                       (False).
    """

    if not timeseries:
        exp_value = -0.5 * \
            T.sum((output - y.dimshuffle('x', 0, 1))**2, axis=2) * precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value),
                                                  -1 * T.min(exp_value)),
                                             T.min(exp_value), T.max(exp_value)
                                             )

        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value),
                                           axis=0)) +
                               max_exp_value)  # -\
        #       self.y.shape[0]*(T.log(self.m)+self.y.shape[1]/2.*T.log(2*np.pi))

    else:
        exp_value = -0.5 * \
            T.sum((output - y.dimshuffle(0, 'x', 1, 2))**2, axis=3) * precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value), -1 * T.min(
            exp_value)),
            T.max(exp_value), T.min(exp_value))

        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value),
                                           axis=1)) +
                               max_exp_value)
    return log_likelihood


def get_no_stochastic_log_likelihood(output, y, precision, timeseries):
    """outputs theano variable that contains the UNNORMALIZED log-likelihood.

    :type output: theano.tensor3 if timeseries False/ theano.tensor4 if
                                                               timeseries True.
    :param output: theano output variable of network.

    :type y: theano.matrix.
    :param y: theano variable containing true output.

    :type precision: float.
    :param precision: precision parameter of gaussian distribution.
                      Useful to overcome numerical problems.
                      If getting nans try to increase its value.

    :type timeseries: bool.
    :param timeseries: tells if the network is recurrent (True) or feedforward
                                                                       (False).
    """

    if not timeseries:
        exp_value = -0.5 * \
            T.sum((output - y)**2, axis=1) * precision

        log_likelihood = T.sum(exp_value)  # -\
        #       self.y.shape[0]*(T.log(self.m)+self.y.shape[1]/2.*T.log(2*np.pi))

    else:
        exp_value = -0.5 * \
            T.sum((output - y)**2, axis=2) * precision
        log_likelihood = T.sum(exp_value)
    return log_likelihood


def get_log_likelihood(output, y, precision, timeseries, stochastic_samples=True):
    """outputs theano variable that contains the UNNORMALIZED log-likelihood.

    :type output: theano.tensor3 if timeseries False/ theano.tensor4 if
                                                               timeseries True.
    :param output: theano output variable of network.

    :type y: theano.matrix.
    :param y: theano variable containing true output.

    :type precision: float.
    :param precision: precision parameter of gaussian distribution.
                      Useful to overcome numerical problems.
                      If getting nans try to increase its value.

    :type timeseries: bool.
    :param timeseries: tells if the network is recurrent (True) or feedforward
                                                                       (False).
    """

    if not timeseries:
        exp_value = -0.5 * \
            T.sum((output - y.dimshuffle('x', 0, 1))**2, axis=2) * precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value),
                                                  -1 * T.min(exp_value)),
                                             T.min(exp_value), T.max(exp_value)
                                             )

        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value),
                                           axis=0)) +
                               max_exp_value)  # -\
        #       self.y.shape[0]*(T.log(self.m)+self.y.shape[1]/2.*T.log(2*np.pi))

    else:
        exp_value = -0.5 * \
            T.sum((output - y.dimshuffle(0, 'x', 1, 2))**2, axis=3) * precision
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value), -1 * T.min(
            exp_value)),
            T.max(exp_value), T.min(exp_value))

        log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value),
                                           axis=1)) +
                               max_exp_value)
    return log_likelihood


def get_weight_init_values(n_in, n_out, activation=None, rng=None,
                           W_values=None):
    """Returns initial weights for a weight matrix.

    :type n_in: int.
    :param n_in: input dimensionality of layer.

    :type n_out: int.
    :param n_out: output dimensionality of layer.

    :type activation: function/theano function.
    :param activation: activation function of layer.

    :type rng: numpy.random.RandomState.
    :param rng: random number generator used to initialize weights.

    :type W_values: numpy.array / None.
    :param W_values: initialization values. If None, random numbers from
                     uniform distribution are drawn.
    """

    if W_values is None:
        if rng is None:
            rng = np.random.RandomState(0)

        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_out, n_in)
            ),
            dtype=theano.config.floatX
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
    else:
        W_values = np.asarray(np.array(W_values), dtype=theano.config.floatX)
    return W_values


def get_bias_init_values(n, b0=None, b_values=None):
    """Returns initial bias values.

    :type n: int.
    :param n: vector size of biases.

    :type b0: float, double.
    :param b0: if b_values is None, bias is initialize to vector b0.
               If None bias initialized to vector of zeros.

    :type b_values: numpy.array, None.
    :param b_values: initialization values. If None, b0 is used.
    """

    if b_values is None:
        if b0 is None:
            b_values = np.zeros((n,), dtype=theano.config.floatX)
        else:
            b_values = np.empty((n,), dtype=theano.config.floatX)
            b_values.fill(b0)
    else:
        b_values = np.asarray(np.array(b_values), dtype=theano.config.floatX)
    return b_values


def init_bn(layer, n_in, gamma_values=None, beta_values=None):
    if gamma_values is None:
        gamma_values = np.ones(n_in, dtype=theano.config.floatX)

    if beta_values is None:
        beta_values = np.zeros(n_in, dtype=theano.config.floatX)

    layer.gamma = theano.shared(
        value=np.asarray(gamma_values, dtype=theano.config.floatX), name='gamma', borrow=True)
    layer.beta = theano.shared(
        value=np.asarray(beta_values, dtype=theano.config.floatX), name='beta', borrow=True)
