import types
from types import IntType
from types import ListType
import numpy as np
import theano
import theano.tensor as T
import logging
import json
from util import parse_activations
from util import get_weight_init_values
from util import get_bias_init_values
from util import get_activation_function

class RNN():
    def define_network(input):
        """To be implemented in subclasses. Sets up the network variables and connections."""
        self.y_pred = None
        raise NotImplementedError
 
class RNNOutputLayer():
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name, W_values=None,
                                                                                    b_values=None):
        """
        RNN output layer.
        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type input_var: theano.tensor.dmatrix.
        :param input_var: a symbolic tensor of shape (m, n_samples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.

        :type activation_name: string.
        :param activation_name: name of activation function.

        :type W_values: numpy.array.
        :param W_values: initialization values of the weights.

        :type b_values: numpy array.
        :param b_values: initialization values of the bias.
        """
        self.input = input_var
        if W_values is None:
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
        if b_values is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)

        W = theano.shared(value=W_values, name='V', borrow=True)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.n_in = n_in
        self.n_out = n_out
        self.W = W
        self.b = b

        self.params = [self.W, self.b]
        self.activation = activation
        self.activation_name = activation_name

        def h_step(x):
            a = T.tensordot(x, self.W, axes=([2,1])) + self.b
            h_t = self.activation(a)
            return h_t
        self.output, _ = theano.scan(h_step, sequences=self.input)
            

class VanillaRNNHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name,
                                W_f_values=None, W_r_values=None, b_values=None, h_values=None):
        """
        Hidden layer: Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out,).
        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type input_var: theano.tensor.dmatrix.
        :param input_var: a symbolic tensor of shape (t, m stochastic draws,
                                                                            training_samples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.

        :type activation_name: string
        :param activation_name: name of activation function.

        :type W_f_values: numpy.array.
        :param W_f_values: initialization values of the forward weights.

        :type W_r_values: numpy.array.
        :param W_r_values: initialization values of the recursive weights.

        :type b_values: numpy array.
        :param b_values: initialization values of the bias.
        """
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        if W_f_values is None:
            W_f_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_in)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_f_values *= 4
        
        if W_r_values is None:
            W_r_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_out + n_out)),
                    high=np.sqrt(6. / (n_out + n_out)),
                    size=(n_out, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_r_values *= 4

        if b_values is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        if h_values is None:
            h_values = np.zeros((n_out,), dtype=theano.config.floatX)
        W_f = theano.shared(value=W_f_values, name='W_f', borrow=True)
        W_r = theano.shared(value=W_r_values, name='W_r', borrow=True)
        b = theano.shared(value=b_values, name='b', borrow=True)
        h0 = theano.shared(value=h_values, name='h0', borrow=True)

        self.W_f = W_f
        self.W_r = W_r
        self.b = b
        self.h0 = h0
        self.params = [self.W_f, self.W_r, self.b, self.h0]

        self.activation = activation

        def h_step(x, h_tm1):
            a = T.tensordot(x, self.W_f, axes=([2,1])) + T.tensordot(h_tm1, self.W_r,
                                                                           axes=[2,1]) + self.b
            h_t = self.activation(a)
            return h_t
        self.output, _ = theano.scan(h_step, sequences=self.input,
                                            outputs_info=T.alloc(self.h0, self.input.shape[1],
                                                                self.input.shape[2],
                                                                self.n_out))
            

class VanillaRNN(RNN):

    def __init__(self, n_in, n_hidden, n_out, activation_list, rng=None, layers_info=None,
                                                                                    input_var=None):
        """Defines the basics of a Vanilla Recurrent Neural Network used on top of a LBN.

        :param n_in: integer defining the number of input units.
        :param n_hidden: list of integers defining the number of hidden units per layer.
        :param n_out: integer defining the number of output units.
        :param activation: list of size len(n_hidden) + 1 defining the activation function per layer.
        :param prng: random number generator.
        """
        if input_var is None:
            self.x = T.tensor3('x', dtype=theano.config.floatX)
        else:
            self.x = input_var
        if rng is None:
            self.rng = np.random.RandomState(0)
        else:
            self.rng = rng

        assert type(n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)

        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)
        self.n_hidden = np.array(n_hidden)

        self.activation_list = activation_list
        assert type(self.activation_list) is ListType, "activation must be a list:\
                                                                {0!r}".format(self.activation_list)

        assert len(n_hidden) + 1 == len(self.activation_list),\
        "Activation list must have len(n_hidden) + 1 values. Activation: {0!r}, n_hidden: \
                                                {1!r}".format(self.activation_list, n_hidden)

        assert type(n_out) is IntType, "n_out must be an int: {0!r}".format(n_out)


        self.defined = False
        self.parse_properties(n_in, n_hidden, n_out, activation_list)
        self.type = 'VanillaRNN'
        self.opt = {'type': self.type, 'n_in': self.n_in, 'n_hidden': self.n_hidden,
                    'n_out': self.n_out, 'activation': self.activation_list}

        logging.info('RNN loaded. Type: {0}, input layer: {1}, hidden layers: {2}, output layer: '
                    '{3}, activation: {4}'.format(self.type, self.n_in, self.n_hidden, self.n_out,
                                                                            self.activation_list))
                          
        self.define_network(layers_info=layers_info)

    def parse_properties(self, n_in, n_hidden, n_out, activation_list):
        self.n_hidden = np.array(n_hidden)
        self.n_out = n_out
        self.n_in = n_in
        self.activation_names = activation_list
        self.activations, self.activations_prime = parse_activations(activation_list)

    def define_network(self, layers_info=None):

        self.hidden_layers = [None]*self.n_hidden.size

        self.params = []

        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = VanillaRNNHiddenLayer(self.rng, self.x, self.n_in, h,
                                                            self.activations[i],
                                                            self.activation_names[i],
                                                            W_f_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['W_f']),
                                                            W_r_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['W_r']),
                                                            b_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['b']),
                                                            h_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['h0']))
                                                            
            else:
                self.hidden_layers[i] = VanillaRNNHiddenLayer(self.rng,
                                                            self.hidden_layers[i-1].output,
                                                            self.n_hidden[i-1], h,
                                                            self.activations[i],
                                                            self.activation_names[i],
                                                            W_f_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['W_f']),
                                                            W_r_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['W_r']),
                                                            b_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['b']),
                                                            h_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['hidden_layers'][i]\
                                                            ['VanillaRNNHiddenLayer']['h0']))

        
            self.params.append(self.hidden_layers[i].params)

        self.output_layer = RNNOutputLayer(self.rng, self.hidden_layers[-1].output,
                                                            self.n_hidden[-1],
                                                            self.n_out,
                                                            self.activations[-1],
                                                            self.activation_names[-1],
                                                            W_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['output_layer']\
                                                            ['RNNOutputLayer']['W']),
                                                            b_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['output_layer']\
                                                            ['RNNOutputLayer']['b']))

        
        self.params.append(self.output_layer.params)
        self.output = self.output_layer.output

    def generate_saving_string(self):
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in":self.n_in, "n_hidden":self.n_hidden.tolist(),
                "n_out":self.n_out,
                "activations":self.activation_names})
        output_string += ", \"layers\": {\"hidden_layers\":["
        for k, l in enumerate(self.hidden_layers):
            if k > 0:
                output_string += ","
            output_string += "{\"VanillaRNNHiddenLayer\":"
            output_string += json.dumps({"n_in": l.n_in, "n_out": l.n_out,
                                        "activation":l.activation_name,
                                        "W_f":l.W_f.get_value().tolist(),
                                        "W_r":l.W_r.get_value().tolist(),
                                        "h0": l.h0.get_value().tolist(),
                                        "b":l.b.get_value().tolist()})
            output_string += "}"
        output_string += "]"
        output_string += ", \"output_layer\":{\"RNNOutputLayer\":"
        l = self.output_layer
        output_string += json.dumps({"n_in": l.n_in, "n_out": l.n_out,
                                        "activation":l.activation_name,
                                        "W":l.W.get_value().tolist(),
                                        "b":l.b.get_value().tolist()})
        output_string += "}}}"

        return output_string

    def save_network(self, fname):
        """
        Saves network to json file.

        :type fname: string.
        :param fname: file name (with local or global path) where to store the network.
        """ 
        output_string = self.generate_saving_string()
        with open('{0}'.format(fname), 'w') as f:
            f.write(output_string)

    @classmethod
    def init_from_file(cls, fname, input_var=None):
        """
        Loads a saved network from file fname.
        :type fname: string.
        :param fname: file name (with local or global path) from where to load the network.
        """
        with open(fname) as f:
            network_description = json.load(f)

        network_properties= network_description['network_properties']
        loaded_lbn = cls(network_properties['n_in'], network_properties['n_hidden'],
                        network_properties['n_out'], network_properties['activations'],
                        layers_info=network_description['layers'],
                        input_var=input_var)
        return loaded_lbn

        

class LSTM(RNN):

    def __init__(self, n_in, n_hidden, n_out, activation_list, rng=None, input_var=None,
                                                                                layers_info=None):
        """Defines the basics of a LSTM Neural Network.

        :param n_in: integer defining the number of input units.
        :param n_hidden: list of integers defining the number of hidden units per layer.
        :param n_out: integer defining the number of output units.
        :param activation: list with activation function for [input gate, candidate gate,
        forget gate, output gate, network output].
        :param rng: random number generator.
        :bias_init: bias initialization.
        """

        if input_var is None:
            self.x = T.tensor3('x', dtype=theano.config.floatX)
        else:
            self.x = input_var

        if rng is None:
            self.rng = np.random.RandomState(0)
        else:
            self.rng = rng

        assert type(n_in) is IntType, "n_in must be an integer: {0!r}".format(self.n_in)
        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)

        assert type(activation_list) is ListType, "activation must be a list: {0!r}".format(
                                                                            self.activation_list)
                                     
        assert len(activation_list) is 2, "activation list must be of length 2: "\
                                                    "[LSTMLayer activations, output activation]"

        assert len(activation_list[0]) is 5, "activation list for LSTMLayer musy be of length 5"
        assert type(n_out) is IntType, "n_out must be an int: {0!r}".format(self.n_out)


        self.defined = False
        self.parse_properties(n_in, n_hidden, n_out, activation_list)

        self.type = 'LSTM'
        self.opt = {'type': self.type, 'n_in': self.n_in, 'n_hidden': self.n_hidden,
                    'n_out': self.n_out, 'activation': self.activation_names}

        logging.info('RNN loaded. Type: {0}, input layer: {1}, hidden layers: {2}, output layer: {3}'
            .format(self.type, self.n_in, self.n_hidden, self.n_out))
            
        self.define_network(layers_info=layers_info)

    def parse_properties(self, n_in, n_hidden, n_out, activation_list):
        self.n_hidden = np.array(n_hidden)
        self.n_out = n_out
        self.n_in = n_in
        self.activation_names = activation_list
        self.activations = [None]*2
        self.activations[0], _ = parse_activations(activation_list[0])
        self.activations[1] = get_activation_function(activation_list[1])


    def define_network(self, layers_info=None):

        self.hidden_layers = [None]*self.n_hidden.size
        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = LSTMHiddenLayer(self.rng, self.x, self.n_in, h,
                                                            self.activations[0],
                                                            self.activation_names[0],
                                                            weights=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['weights'],
                                                            biases=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['biases'],
                                                            zero_values=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['zero_values'])
                                                            
            else:
                self.hidden_layers[i] = LSTMHiddenLayer(self.rng,
                                                            self.hidden_layers[i-1].output,
                                                            self.n_hidden[i-1], h,
                                                            self.activations[0],
                                                            self.activation_names[0],
                                                            weights=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['weights'],
                                                            biases=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['biases'],
                                                            zero_values=None if layers_info is None
                                                            else layers_info['hidden_layers'][i]\
                                                            ['LSTMHiddenLayer']['zero_values'])

            self.params.append(self.hidden_layers[i].params)
        self.output_layer = RNNOutputLayer(self.rng, self.hidden_layers[-1].output,
                                                            self.n_hidden[-1],
                                                            self.n_out,
                                                            self.activations[-1],
                                                            self.activation_names[-1],
                                                            W_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['output_layer']\
                                                            ['LSTMHiddenLayer']['W']),
                                                            b_values=None if layers_info is None
                                                            else np.array(
                                                            layers_info['output_layer']\
                                                            ['LSTMHiddenLayer']['b']))

        self.params.append(self.output_layer.params)
        self.output = self.output_layer.output


    def generate_saving_string(self):
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in":self.n_in, "n_hidden":self.n_hidden.tolist(),
                "n_out":self.n_out,
                "activations":self.activation_names})
        output_string += ", \"layers\": {\"hidden_layers\":["
        for k, l in enumerate(self.hidden_layers):
            if k > 0:
                output_string += ","
            output_string += "{\"LSTMHiddenLayer\":"
            weights = {"Whi": l.Whi.get_value().toList(),
                        "Whj": l.Whj.get_value().toList(),
                        "Whf": l.Whf.get_value().toList(),
                        "Who": l.Who.get_value().toList(),
                        "Wxi": l.Wxi.get_value().toList(),
                        "Wxj": l.Wxj.get_value().toList(),
                        "Wxo": l.Wxo.get_value().toList()}

            biases = {  "bi": l.bi.get_value().toList(),
                        "bj": l.bj.get_value().toList(),
                        "bf": l.bf.get_value().toList(),
                        "bo": l.bo.get_value().toList()}

            zero_values = {"h0": l.h0.get_value().toList(),
                            "c0": l.c0.get_value().toList()}

            output_string += json.dumps({"n_in": l.n_in, "n_out": l.n_out,
                                        "activation":l.activation_names,
                                        "weights": weights,
                                        "biases": biases,
                                        "zero_values": zero_values})

            output_string += "}"
        output_string += "]"
        output_string += ", \"output_layer\":{\"RNNOutputLayer\":"
        l = self.output_layer
        output_string += json.dumps({"n_in": l.n_in, "n_out": l.n_out,
                                        "activation":l.activation_name,
                                        "W":l.W.get_value().tolist(),
                                        "b":l.b.get_value().tolist()})
        output_string += "}}}"

        return output_string

    def save_network(self, fname):
        """
        Saves network to json file.

        :type fname: string.
        :param fname: file name (with local or global path) where to store the network.
        """ 
        output_string = self.generate_saving_string()
        with open('{0}'.format(fname), 'w') as f:
            f.write(output_string)

    @classmethod
    def init_from_file(cls, fname, input_var=None):
        """
        Loads a saved network from file fname.
        :type fname: string.
        :param fname: file name (with local or global path) from where to load the network.
        """
        with open(fname) as f:
            network_description = json.load(f)

        network_properties= network_description['network_properties']
        loaded_lbn = cls(network_properties['n_in'], network_properties['n_hidden'],
                        network_properties['n_out'], network_properties['activations'],
                        layers_info=network_description['layers'],
                        input_var=input_var)
        return loaded_lbn



class LSTMHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activations, activation_names, weights=None,
                                                                                biases=None,
                                                                                zero_values=None):
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.activation_names = activation_names
        self.rng=rng

        if weights is None:
            Whi_values = get_weight_init_values(n_out, n_out, activation=activations[0], rng=self.rng)
            Whj_values = get_weight_init_values(n_out, n_out, activation=activations[1], rng=self.rng)
            Whf_values = get_weight_init_values(n_out, n_out, activation=activations[2], rng=self.rng)
            Who_values = get_weight_init_values(n_out, n_out, activation=activations[3], rng=self.rng)
            Wxi_values = get_weight_init_values(n_in, n_out, activation=activations[0], rng=self.rng)
            Wxj_values = get_weight_init_values(n_in, n_out, activation=activations[1], rng=self.rng)
            Wxf_values = get_weight_init_values(n_in, n_out, activation=activations[2], rng=self.rng)
            Wxo_values = get_weight_init_values(n_in, n_out, activation=activations[3], rng=self.rng)

        else:
            Whi_values = weights['Whi']
            Whj_values = weights['Whj']
            Whf_values = weights['Whf']
            Who_values = weights['Who']
            Wxi_values = weights['Wxi']
            Wxj_values = weights['Wxj']
            Wxf_values = weights['Wxf']
            Wxo_values = weights['Wxo']

        if biases is None:

            bi_values = get_bias_init_values(n_out)
            bj_values = get_bias_init_values(n_out)
            bf_values = get_bias_init_values(n_out)
            bo_values = get_bias_init_values(n_out)
        else:

            bi_values = biases['bi']
            bj_values = biases['bj']
            bf_values = biases['bf']
            bo_values = biases['bo']

        if zero_values is None:
            h0_values = get_bias_init_values(n_out)
            c0_values = get_bias_init_values(n_out)
        else:
            h0_values = zero_values['h0']
            c0_values = zero_values['c0']

        self.Whi = theano.shared(value=Whi_values, name='Whi', borrow=True)
        self.Whj = theano.shared(value=Whj_values, name='Whj', borrow=True)
        self.Whf = theano.shared(value=Whf_values, name='Whf', borrow=True)
        self.Who = theano.shared(value=Who_values, name='Who', borrow=True)
        self.Wxi = theano.shared(value=Wxi_values, name='Wxi', borrow=True)
        self.Wxj = theano.shared(value=Wxj_values, name='Wxj', borrow=True)
        self.Wxf = theano.shared(value=Wxj_values, name='Wxf', borrow=True)
        self.Wxo = theano.shared(value=Wxo_values, name='Wxo', borrow=True)

        self.h0 = theano.shared(value=h0_values, name='ho', borrow=True)
        self.c0 = theano.shared(value=c0_values, name='co', borrow=True)
        self.bi = theano.shared(value=bi_values, name='bi', borrow=True)
        self.bj = theano.shared(value=bj_values, name='bj', borrow=True)
        self.bf = theano.shared(value=bf_values, name='bf', borrow=True)
        self.bo = theano.shared(value=bo_values, name='bo', borrow=True)

        self.params = [self.Whi, self.Whj, self.Whf, self.Who, self.Wxi, self.Wxj, self.Wxo, self.Wxf,
                                    self.h0, self.c0, self.bi, self.bj, self.bf, self.bo]


        self.activations = activations
        def h_step(x_t, h_tm1, c_tm1):
            a_input_gate = T.tensordot(x_t, self.Wxi, axes=([2,1])) + T.tensordot(h_tm1, self.Whi,
                                                                           axes=[2,1]) + self.bi
            input_gate = self.activations[0](a_input_gate)

            a_candidate_gate = T.tensordot(x_t, self.Wxj, axes=([2,1])) + T.tensordot(h_tm1,
                                                                    self.Whj, axes=[2,1]) + self.bj
            candidate_gate = self.activations[1](a_candidate_gate)
            a_forget_gate = T.tensordot(x_t, self.Wxf, axes=([2,1])) + T.tensordot(h_tm1, self.Whf,
                                                                           axes=[2,1]) + self.bf
            forget_gate = self.activations[2](a_forget_gate)

            a_output_gate = T.tensordot(x_t, self.Wxo, axes=([2,1])) + T.tensordot(h_tm1, self.Who,
                                                                           axes=[2,1]) + self.bo
            output_gate = self.activations[3](a_output_gate)
            
            c_t = c_tm1*forget_gate+input_gate*candidate_gate
            a_t = c_t*output_gate
            h_t = self.activations[4](a_t)*output_gate
            return h_t, c_t

        [self.output, c_t], _ = theano.scan(h_step, sequences=self.input,
                                                outputs_info=[T.alloc(self.h0, self.input.shape[1],
                                                        self.input.shape[2], self.n_out),
                                                    T.alloc(self.c0, self.input.shape[1],
                                                        self.input.shape[2], self.n_out),])

