import theano
import theano.tensor as T
import numpy as np
from types import IntType
from types import ListType
import json
from util import parse_activations
from util import get_weight_init_values
from util import get_bias_init_values
from util import init_bn
from util import get_correlated_weights_init_values
from util import get_correlated_biases_init_values


class CorrelatedLayer(object):

    def __init__(self, input_var, n_in, n_out, activation_name, activation, rng=None,
                 W_values=None, b_values=None, W_correlated_values=None, b_correlated_values=None,
                 timeseries_layer=False,
                 epsilon=1e-12):
        """
        Hidden layer: Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out,).

        :type input_var: theano.tensor.dmatrix, theano.tensor.tensor3 or theano.tensor.tensor4.
        :param input_var: a symbolic tensor of shape (n_samples, n_in) or (m, n_samples, n_in).
                            If timeseries_layer=False then (t_points, n_samples, n_in) or
                            (t_points, m, n_samples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation_name: string
        :param activation_name: name of activation function.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.

        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type W_values: numpy.array.
        :param W_values: initialization values of the weights.

        :type b_values: numpy array.
        :param b_values: initialization values of the bias.

        :type timeseries_layer: bool.
        :param timeseries_layer: if True the input is considered to be timeseries.
        """

        self.input = input_var

        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        self.activation = activation

        W_correlated_values = get_correlated_weights_init_values(
            n_in, n_out, activation=activation, rng=rng, W_correlated_values=W_correlated_values,
            activation_name=self.activation_name)

        W_correlated = [theano.shared(
            value=wi, name='W{0}_corr', borrow=True) for i, wi in enumerate(W_correlated_values)]

        b_correlated_values = get_correlated_biases_init_values(
            n_out, b_correlated_values=b_correlated_values)

        b_correlated = [theano.shared(
            value=bi, name='b{0}_corr', borrow=True) for i, bi in enumerate(b_correlated_values)]

        W_values = get_weight_init_values(
            n_in, n_out, activation=activation, rng=rng, W_values=W_values,
            activation_name=self.activation_name)

        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = get_bias_init_values(n_out, b_values=b_values)

        b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.W_correlated = W_correlated
        self.b_correlated = b_correlated
        self.params = [self.W, self.b] + self.W_correlated + self.b_correlated
        self.timeseries = timeseries_layer

        if self.timeseries:
            self.timeseries_layer()
        else:
            self.feedforward_layer()

    def feedforward_layer(self):
        self.a = T.dot(self.input, self.W.T) + self.b
        for i in xrange(1, self.n_out):
            self.a = T.set_subtensor(self.a[i], T.dot(
                self.a[:i], self.W_correlated[i - 1])) + self.b_correlated[i - 1]
        self.output = self.activation(self.a)

    def timeseries_layer(self):
        def step(x):
            a = T.dot(x, self.W.T) + self.b
            for i in xrange(1, self.n_out):
                a = T.set_subtensor(a[i], T.dot(
                    a[:i], self.W_correlated[i - 1])) + self.b_correlated[i - 1]
            return a
        self.a, _ = theano.scan(step, sequences=self.input)
        self.output = self.activation(self.a)


class HiddenLayer(object):

    def __init__(self, input_var, n_in, n_out, activation_name, activation, rng=None,
                 W_values=None, b_values=None, timeseries_layer=False,
                 batch_normalization=False, gamma_values=None, beta_values=None, epsilon=1e-12,
                 fixed_means=False, stdb=None, mub=None, dropout=False, training=None):
        """
        Hidden layer: Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out,).

        :type input_var: theano.tensor.dmatrix, theano.tensor.tensor3 or theano.tensor.tensor4.
        :param input_var: a symbolic tensor of shape (n_samples, n_in) or (m, n_samples, n_in).
                            If timeseries_layer=False then (t_points, n_samples, n_in) or
                            (t_points, m, n_samples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation_name: string
        :param activation_name: name of activation function.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.

        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type W_values: numpy.array.
        :param W_values: initialization values of the weights.

        :type b_values: numpy array.
        :param b_values: initialization values of the bias.

        :type timeseries_layer: bool.
        :param timeseries_layer: if True the input is considered to be timeseries.
        """
        if fixed_means:
            assert type(
                stdb) is np.ndarray, "Minibatch std must be a numpy array. Given: {0!r}".format(stdb)
            assert stdb.size == n_out, "Minibatch std must be of size n_out ({0}). Given shape: {1}".format(
                n_out, stdb.shape)

            assert type(
                mub) is np.ndarray, "Minibatch mean must be a numpy array. Given: {0!r}".format(mub)
            assert mub.size == n_out, "Minibatch mean must be of size n_out ({0}). Given shape: {1}".format(
                n_out, mub.shape)

        self.input = input_var
        self.dropout = dropout
        if self.dropout:
            assert training is not None
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        self.activation = activation
        W_values = get_weight_init_values(
            n_in, n_out, activation=activation, rng=rng, W_values=W_values,
            activation_name=self.activation_name)

        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = get_bias_init_values(n_out, b_values=b_values)

        b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.timeseries = timeseries_layer

        self.batch_normalization = batch_normalization

        if self.batch_normalization:
            init_bn(self, n_out, gamma_values=gamma_values,
                    beta_values=beta_values)
            self.fixed_means = fixed_means
            self.mub = None
            self.stdb = None
            self.epsilon = epsilon
            self.params.append(self.gamma)
            self.params.append(self.beta)

        if self.timeseries:
            self.timeseries_layer()
        else:
            self.feedforward_layer()

        trng = theano.tensor.shared_randomstreams.RandomStreams(1234)

        if self.dropout:
            self.output = theano.ifelse.ifelse(training,  T.switch(
                trng.binomial(size=self.output.shape, p=0.5), self.output, 0),  0.5 * self.output)

    def feedforward_layer(self):
        self.a = T.dot(self.input, self.W.T) + self.b
        if self.batch_normalization:
            if self.fixed_means:
                mub = self.mub
                stdb = self.stdb
            else:
                mub = T.mean(self.a, axis=0)
                sigma2b = T.var(self.a, axis=0)
                stdb = T.sqrt(sigma2b + self.epsilon)

            self.a = theano.tensor.nnet.bn.batch_normalization(
                self.a, self.gamma, self.beta, mub, stdb)

        self.output = self.activation(self.a)

    def timeseries_layer(self):
        def step(x):
            a = T.dot(x, self.W.T) + self.b
            return a
        self.a, _ = theano.scan(step, sequences=self.input)
        if self.batch_normalization:
            if self.fixed_means:
                mub = self.mub
                stdb = self.stdb
            else:
                mub = T.sum(self.a, axis=(0, 1)) * 1. / \
                    T.prod(self.a.shape[:-1])

                sigma2b = T.sum((self.a - mub)**2, axis=(0, 1)) * \
                    1. / T.prod(self.a.shape[:-1])

                stdb = T.sqrt(sigma2b + self.epsilon)

            self.a = theano.tensor.nnet.bn.batch_normalization(
                self.a, self.gamma, self.beta, mub, stdb)

        self.output = self.activation(self.a)


class MLPLayer(object):

    def __init__(self, n_in, n_hidden, activation_names, timeseries_network=False,
                 input_var=None,
                 layers_info=None,
                 batch_normalization=False,
                 dropout=False, training=None):
        """
        MLP network used as a layer in a bigger network.

        :type n_in: integer.
        :param n_in: input dimensionality.

        :type n_hidden: list of integers.
        :param n_hidden: number of hidden units per layer.

        :type activation_names: list of strings.
        :param activation_name: name of activation functions.

        :type timeseries_layer: bool.
        :param timeseries_layer: if True the input is considered to be timeseries.

        :type input_var: theano.tensor.dmatrix, theano.tensor.tensor3 or theano.tensor.tensor4.
        :param input_var: a symbolic tensor of shape (n_samples, n_in) or (m, n_samples, n_in).
                            If timeseries_layer=False then (t_points, n_samples, n_in) or
                            (t_points, m, n_samples, n_in).

        :type layers_info: dictionary or None.
        :param layers_info: network information.

        :type dropout: bool.
        :param dropout: sets the use of dropout or not in the network.

        :type training: theano.tensor.scalar or None.
        :param training: only available if dropout is True. A theano input variable that tells if the
        networks is being used in training or predicting format. If None and dropout is True a
        theano.tensor.scalar variable is created.
        """

        if input_var is None:
            if timeseries_network:
                self.x = T.tensor3('x', dtype=theano.config.floatX)
                self.y = T.tensor3('y', dtype=theano.config.floatX)
            else:
                self.x = T.matrix('x', dtype=theano.config.floatX)
                self.y = T.matrix('y', dtype=theano.config.floatX)
        else:
            self.x = input_var

        self.training = training

        self.timeseries_network = timeseries_network
        self.rng = np.random.RandomState()
        self.parse_properties(
            n_in, n_hidden, activation_names, batch_normalization, dropout)

        self.define_network(layers_info=layers_info)

    def parse_properties(self, n_in, n_hidden, activation_names, batch_normalization, dropout):
        """
        :type n_in: integer.
        :param n_in: input dimensionality.

        :type n_hidden: list of integers.
        :param n_hidden: number of hidden units per layer.

        :type activation_names: list of strings.
        :param activation_name: name of activation functions.
        """

        assert type(
            n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(
            n_hidden)
        assert type(activation_names) is ListType, "activation_names must be a list: {0!r}".format(
            activation_names)
        assert len(n_hidden) == len(
            activation_names), "len(n_hidden) must be =="\
            " len(det_activations). n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden,
                                                                                       activation_names)
        assert type(batch_normalization) is bool, "batch_normalization must be bool. Given: {0!r}".format(
            batch_normalization)
        assert type(dropout) is bool

        self.dropout = dropout

        if self.dropout and self.training is None:
            self.training = T.scalar('scalar')

        self.n_in = n_in
        self.n_hidden = np.array(n_hidden)
        self.activation_names = activation_names
        self.activation, self.activation_prime = parse_activations(
            activation_names)
        self.batch_normalization = batch_normalization

    def define_network(self, layers_info=None):
        """
        :type layers_info: dictionary or None.
        :param layers_info: network information.
        """
        self.hidden_layers = [None] * self.n_hidden.size
        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = HiddenLayer(self.x, self.n_in, h, self.activation_names[i],
                                                    self.activation[
                    i], rng=self.rng,
                    W_values=None if layers_info is None else
                    np.array(
                    layers_info['layers'][
                        i]['HiddenLayer']
                    ['W']),
                    b_values=None if layers_info is None else
                    np.array(
                    layers_info['layers'][
                        i]['HiddenLayer']
                    ['b']),
                    timeseries_layer=self.timeseries_network,
                    batch_normalization=self.batch_normalization,
                    gamma_values=None if layers_info is None or
                    'gamma_values' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['gamma_values'],
                    beta_values=None if layers_info is None or 'beta_values' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['beta_values'],
                    epsilon=1e-12 if layers_info is None or 'epsilon' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['epsilon'],
                    fixed_means=False, training=self.training, dropout=self.dropout)
            else:
                self.hidden_layers[i] = HiddenLayer(self.hidden_layers[i - 1].output,
                                                    self.n_hidden[
                    i - 1], h, self.activation_names[i],
                    self.activation[
                    i], rng=self.rng,
                    W_values=None if layers_info is None else
                    np.array(
                    layers_info['layers'][
                        i]['HiddenLayer']
                    ['W']),
                    b_values=None if layers_info is None else
                    np.array(
                    layers_info['layers'][
                        i]['HiddenLayer']
                    ['b']),
                    timeseries_layer=self.timeseries_network,
                    batch_normalization=self.batch_normalization,
                    gamma_values=None if layers_info is None or 'gamma_values' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['gamma_values'],
                    beta_values=None if layers_info is None or 'beta_values' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['beta_values'],
                    epsilon=1e-12 if layers_info is None or 'epsilon' not in layers_info[
                    'layers'][i]['HiddenLayer'].keys() else layers_info['layers'][i]['HiddenLayer']['epsilon'],
                    fixed_means=False, training=self.training, dropout=self.dropout)

            self.params.append(self.hidden_layers[i].params)

        self.output = self.hidden_layers[-1].output

        if self.dropout:
            givens_dict = {self.training: np.float64(
                0).astype(theano.config.floatX)}
        else:
            givens_dict = {}

        self.predict = theano.function(
            inputs=[self.x], outputs=self.output,
            givens=givens_dict)

        self.regulizer_L2 = T.zeros(1)
        self.regulizer_L1 = T.zeros(1)
        for l in self.params:
            for p in l:
                self.regulizer_L2 += (p**2).sum()
                self.regulizer_L1 += p.sum()

    def generate_saving_string(self):
        """ String generated for saving the network"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_hidden": list(self.n_hidden),
                                     "activation": self.activation_names,
                                     "timeseries": self.timeseries_network,
                                     "batch_normalization": self.batch_normalization,
                                     "dropout": self.dropout})

        output_string += ", \"layers\":["
        for j, layer in enumerate(self.hidden_layers):
            if j > 0:
                output_string += ","
            output_string += "{\"HiddenLayer\":"

            buffer_dict = {"n_in": layer.n_in, "n_out": layer.n_out,
                           "activation": layer.activation_name,
                           "W": layer.W.get_value().tolist(),
                           "b": layer.b.get_value().tolist(),
                           "timeseries": layer.timeseries}

            if self.batch_normalization:
                buffer_dict['gamma_values'] = layer.gamma.get_value().tolist()
                buffer_dict['beta_values'] = layer.beta.get_value().tolist()
                buffer_dict['epsilon'] = layer.epsilon

            output_string += json.dumps(buffer_dict)

            output_string += "}"
        output_string += "]}"
        return output_string
