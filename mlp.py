import theano
import theano.tensor as T
import numpy as np
import types
from types import IntType
from types import ListType
from types import FloatType
import json
from util import parse_activations


class HiddenLayer(object):
    
    def __init__(self, input_var, n_in, n_out, activation_name, activation, rng=None, W_values=None,
                                                            b_values=None, timeseries_layer=False):
        
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        self.activation = activation
        if W_values is None:
            if rng is None:
                rng = np.random.RandomState()
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

        W = theano.shared(value=W_values, name='W', borrow=True)
        if b_values is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)

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
        [self.a, self.output] = theano.scan(step, sequences=self.input)


class MLPLayer(object):

    def __init__(self, n_in, n_hidden, activation_names, timeseries_network=False,
                                                                input_var=None,
                                                                layers_info=None):

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
        assert type(n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)
        assert type(activation_names) is ListType, "activation_names must be a list: {0!r}".format(
                                                                                activation_names)
        assert len(n_hidden) == len(activation_names), "len(n_hidden) must be =="\
        " len(det_activations) - 1. n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden, 
                                                                                  activation_names)
        
        
        self.parse_properties(n_in, n_hidden, activation_names)
                               
        self.define_network(layers_info=layers_info)

    def parse_properties(self, n_in, n_hidden, activation_names):
        self.n_in = n_in
        self.n_hidden = np.array(n_hidden)
        self.activation_names = activation_names
        self.activation, self.activation_prime = parse_activations(activation_names)



    def define_network(self, layers_info=None):

        self.hidden_layers = [None]*self.n_hidden.size
        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i==0:
                self.hidden_layers[i] = HiddenLayer(self.x, self.n_in, h, self.activation_names[i],
                                                    self.activation[i], rng=self.rng,
                                                    W_values=None if layers_info is None else 
                                                    np.array(
                                                    layers_info['layers'][i]['HiddenLayer']\
                                                                                ['W']),
                                                    b_values=None if layers_info is None else 
                                                    np.array(
                                                    layers_info['layers'][i]['HiddenLayer']\
                                                                                ['b']),
                                                    timeseries_layer=self.timeseries_network)
            else:
                self.hidden_layers[i] = HiddenLayer(self.hidden_layers[i-1].output,
                                                    self.n_hidden[i-1], h, self.activation_names[i],
                                                    self.activation[i], rng=self.rng,
                                                    W_values=None if layers_info is None else 
                                                    np.array(
                                                    layers_info['layers'][i]['HiddenLayer']\
                                                                                ['W']),
                                                    b_values=None if layers_info is None else 
                                                    np.array(
                                                    layers_info['layers'][i]['HiddenLayer']\
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
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in":self.n_in, "n_hidden": list(self.n_hidden),
                                        "activation": self.activation_names,
                                        "timeseries": self.timeseries_network})
            
        output_string += ", \"layers\":["
        for j, layer in enumerate(self.hidden_layers):
            output_string += "{\"HiddenLayer\":"

            if j > 0:
                output_string += ","
            output_string += json.dumps({"n_in": layer.n_in, "n_out":layer.n_out,
                                        "activation": layer.activation_name,
                                        "W": layer.W.get_value().tolist(),
                                        "b": layer.b.get_value().tolist(),
                                        "timeseries": layer.timeseries})
        
            output_string += "}"
        output_string += "]}"
        return output_string
