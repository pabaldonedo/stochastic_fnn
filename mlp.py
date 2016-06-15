import theano
import theano.tensor as T
import numpy as np
from types import IntType
from types import ListType
import json
from util import parse_activations
from util import get_weight_init_values
from util import get_bias_init_values


class HiddenLayer(object):

    def __init__(self, input_var, n_in, n_out, activation_name, activation, rng=None,
                 W_values=None, b_values=None, timeseries_layer=False):
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
        W_values = get_weight_init_values(
            n_in, n_out, activation=activation, rng=rng, W_values=W_values)

        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = get_bias_init_values(n_out, b_values=b_values)

        b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.timeseries = timeseries_layer

        if self.timeseries:
            self.timeseries_layer()
        else:
            self.feedforward_layer()

    def feedforward_layer(self):
        self.a = T.dot(self.input, self.W.T) + self.b
        self.output = self.activation(self.a)

    def timeseries_layer(self):
        def step(x):
            a = T.dot(x, self.W.T) + self.b
            output = self.activation(a)
            return a, output
        [self.a, self.output], _ = theano.scan(step, sequences=self.input)


class MLPLayer(object):

    def __init__(self, n_in, n_hidden, activation_names, timeseries_network=False,
                 input_var=None,
                 layers_info=None):
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

        self.timeseries_network = timeseries_network
        self.rng = np.random.RandomState()
        self.parse_properties(n_in, n_hidden, activation_names)

        self.define_network(layers_info=layers_info)

    def parse_properties(self, n_in, n_hidden, activation_names):
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
        assert len(n_hidden) == len(activation_names), "len(n_hidden) must be =="\
            " len(det_activations) - 1. n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden,
                                                                                           activation_names)

        self.n_in = n_in
        self.n_hidden = np.array(n_hidden)
        self.activation_names = activation_names
        self.activation, self.activation_prime = parse_activations(
            activation_names)

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
                                                    timeseries_layer=self.timeseries_network)
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
                                                    timeseries_layer=self.timeseries_network)

            self.params.append(self.hidden_layers[i].params)

        self.output = self.hidden_layers[-1].output
        self.predict = theano.function(inputs=[self.x], outputs=self.output)
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
                                     "timeseries": self.timeseries_network})

        output_string += ", \"layers\":["
        for j, layer in enumerate(self.hidden_layers):
            output_string += "{\"HiddenLayer\":"

            if j > 0:
                output_string += ","
            output_string += json.dumps({"n_in": layer.n_in, "n_out": layer.n_out,
                                         "activation": layer.activation_name,
                                         "W": layer.W.get_value().tolist(),
                                         "b": layer.b.get_value().tolist(),
                                         "timeseries": layer.timeseries})

            output_string += "}"
        output_string += "]}"
        return output_string
