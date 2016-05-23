import types
from types import IntType
from types import ListType
import numpy as np
import theano
import theano.tensor as T
import logging
import json
from util import parse_activations


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

        self.activation_list = activation_list
        assert type(self.activation_list) is ListType, "activation must be a list: {0!r}".format(
                                                                            self.activation_list)
                                     
        assert type(n_out) is IntType, "n_out must be an int: {0!r}".format(self.n_out)
        self.type = 'LSTM'

        self.bias_init = bias_init
        assert type(self.bias_init) is ListType, "activation must be a list: {0!r}".format(
                                                                            self.bias_init)

        assert len(bias_init) is 5, "activation initialization list must have lenght 5: [input gate, candidate gate, forget gate, output gate, network output]: {0!r}".format(self.bias_init)

        self.activation, self.activations_prime = parse_activations(self.activation_list)
        self.opt = {'type': self.type, 'n_in': self.n_in, 'n_hidden': self.n_hidden,
                    'n_out': self.n_out, 'activation': self.activation_list,
                    'bias_init': self.bias_init}
        self.defined = False
        self.parse_properties(n_in, n_hidden, n_out, activation_list)
        logging.info('RNN loaded. Type: {0}, input layer: {1}, hidden layers: {2}, output layer: {3}'
            .format(self.type, self.n_in, self.n_hidden, self.n_out))

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
                                        "zero_values": zero_values)

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
         [input gate, candidate gate,
            forget gate, output gate, network output].
        if weights is None:
            Whi_values = get_weight_init_values(n_out, n_out, activation=activations[0])
            Whj_values = get_weight_init_values(n_out, n_out, activation=activations[1])
            Whf_values = get_weight_init_values(n_out, n_out, activation=activations[2])
            Who_values = get_weight_init_values(n_out, n_out, activation=activations[3])
            Wxi_values = get_weight_init_values(n_in, n_out, activation=activations[0])
            Wxj_values = get_weight_init_values(n_in, n_out, activation=activations[1])
            Wxf_values = get_weight_init_values(n_in, n_out, activation=activations[2])
            Wxo_values = get_weight_init_values(n_in, n_out, activation=activations[3])

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
        self.Whxi = theano.shared(value=Wxi_values, name='Wxi', borrow=True)
        self.Whxj = theano.shared(value=Wxj_values, name='Wxj', borrow=True)
        self.Whxo = theano.shared(value=Wxo_values, name='Wxo', borrow=True)

        self.h0 = theano.shared(value=h0_values, name='ho', borrow=True)
        self.c0 = theano.shared(value=c0_values, name='co', borrow=True)
        self.bi = theano.shared(value=bi_values, name='bi', borrow=True)
        self.bj = theano.shared(value=bj_values, name='bj', borrow=True)
        self.bf = theano.shared(value=bf_values, name='bf', borrow=True)
        self.b0 = theano.shared(value=b0_values, name='bo', borrow=True)

        self.params = [self.Whi, self.Whj, self.Whf, self.Who, self.Whxi, self.Whxj, self.Whxo,
                                    self.h0, self.c0, self.bi, self.bj, self.bf, self.b0_values]


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
            output_gate = self.activations[3](output_gate)
            
            c_t = c_tm1*forget_gate+input_gate*candidate_gate
            a_t = c_t*output_gate
            h_t = self.activations[4](a_t)*output_gate
            return h_t, c_t

        self.output, _ = theano.scan(h_step, sequences=self.input,
                                    outputs_info=[T.alloc(self.h0, self.input.shape[1],
                                                        self.input.shape[2], self.n_out),
                                                    T.alloc(self.c0, self.input.shape[1],
                                                        self.input.shape[2], self.n_out),])



    def initialize_input_gate_recurrent_weights(self, param_idx):
        self.Whi = []
        Whi_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            # recurrent weights as a shared variable
            self.Whi.append(self.theta[param_idx:(param_idx + h_units ** 2)].reshape(
                (h_units, h_units)))
            self.Whi[i].name = 'Whi_{0}'.format(i)
            Whi_init = np.append(Whi_init, np.asarray(self.prng.uniform(size=(h_units, h_units),
                                                  low=-0.01, high=0.01),
                                                  dtype=theano.config.floatX).flatten())
            param_idx += h_units ** 2
        
        return param_idx, Whi_init

    def initialize_candidate_state_recurrent_weights(self, param_idx):

        self.Whj = []
        Whj_init = np.empty(0)
        for i, h_units in enumerate(self.n_hidden):
            # recurrent weights as a shared variable
            self.Whj.append(self.theta[param_idx:(param_idx + h_units ** 2)].reshape(
                (h_units, h_units)))
            self.Whj[i].name = 'Whj_{0}'.format(i)
            Whj_init = np.append(Whj_init, np.asarray(self.prng.uniform(size=(h_units, h_units),
                                                  low=-0.01, high=0.01),
                                                  dtype=theano.config.floatX).flatten())
            param_idx += h_units ** 2

        return param_idx, Whj_init

    def initialize_forget_gate_recurrent_weights(self, param_idx):

        self.Whf = []
        Whf_init = np.empty(0)
        for i, h_units in enumerate(self.n_hidden):
            # recurrent weights as a shared variable
            self.Whf.append(self.theta[param_idx:(param_idx + h_units ** 2)].reshape(
                (h_units, h_units)))
            self.Whf[i].name = 'Whf_{0}'.format(i)
            Whf_init = np.append(Whf_init, np.asarray(self.prng.uniform(size=(h_units, h_units),
                                                  low=-0.01, high=0.01),
                                                  dtype=theano.config.floatX).flatten())
            param_idx += h_units ** 2

        return param_idx, Whf_init

    def initialize_output_gate_recurrent_weights(self, param_idx):

        self.Who = []
        Who_init = np.empty(0)
        for i, h_units in enumerate(self.n_hidden):
            # recurrent weights as a shared variable
            self.Who.append(self.theta[param_idx:(param_idx + h_units ** 2)].reshape(
                (h_units, h_units)))
            self.Who[i].name = 'Who_{0}'.format(i)
            Who_init = np.append(Who_init, np.asarray(self.prng.uniform(size=(h_units, h_units),
                                                  low=-0.01, high=0.01),
                                                  dtype=theano.config.floatX).flatten())
            param_idx += h_units ** 2
       
        return param_idx, Who_init

    def initialize_input_gate_forward_weights(self, param_idx):
        self.Wxi = []
        Wxi_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            if i == 0:
                self.Wxi.append(self.theta[param_idx:(param_idx + self.n_in * \
                                          h_units)].reshape((self.n_in, h_units)))
                Wxi_init = np.append(Wxi_init,
                                                np.asarray(self.prng.uniform(size=(self.n_in, h_units),
                                                low=-0.01, high=0.01),
                                                dtype=theano.config.floatX))
                param_idx += self.n_in * self.n_hidden[0]

            else:
                self.Wxi.append(self.theta[param_idx:(param_idx + self.n_hidden[i-1] * \
                                          h_units)].reshape((self.n_hidden[i-1], h_units)))
        
                Wxi_init = np.append(Wxi_init,
                                        np.asarray(self.prng.uniform(size=(self.n_hidden[i-1], h_units),
                                                    low=-0.01, high=0.01),
                                                    dtype=theano.config.floatX))

                param_idx += self.n_hidden[i-1] * h_units

            self.Wxi[i].name = 'Wxi_{0}'.format(i)

        return param_idx, Wxi_init


    def initialize_candidate_state_forward_weights(self, param_idx):

        # same timestamp layers weights
        self.Wxj = []
        Wxj_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            if i == 0:
                self.Wxj.append(self.theta[param_idx:(param_idx + self.n_in * \
                                          h_units)].reshape((self.n_in, h_units)))
                Wxj_init = np.append(Wxj_init,
                                                np.asarray(self.prng.uniform(size=(self.n_in, h_units),
                                                low=-0.01, high=0.01),
                                                dtype=theano.config.floatX))
                param_idx += self.n_in * self.n_hidden[0]

            else:
                self.Wxj.append(self.theta[param_idx:(param_idx + self.n_hidden[i-1] * \
                                          h_units)].reshape((self.n_hidden[i-1], h_units)))
        
                Wxj_init = np.append(Wxj_init,
                                        np.asarray(self.prng.uniform(size=(self.n_hidden[i-1], h_units),
                                                    low=-0.01, high=0.01),
                                                    dtype=theano.config.floatX))

                param_idx += self.n_hidden[i-1] * h_units

            self.Wxj[i].name = 'Wxj_{0}'.format(i)

        return param_idx, Wxj_init

    def initialize_forget_gate_forward_weights(self, param_idx):
        self.Wxf = []
        Wxf_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            if i == 0:
                self.Wxf.append(self.theta[param_idx:(param_idx + self.n_in * \
                                          h_units)].reshape((self.n_in, h_units)))
                Wxf_init = np.append(Wxf_init,
                                                np.asarray(self.prng.uniform(size=(self.n_in, h_units),
                                                low=-0.01, high=0.01),
                                                dtype=theano.config.floatX))
                param_idx += self.n_in * self.n_hidden[0]

            else:
                self.Wxf.append(self.theta[param_idx:(param_idx + self.n_hidden[i-1] * \
                                          h_units)].reshape((self.n_hidden[i-1], h_units)))
        
                Wxf_init = np.append(Wxf_init,
                                        np.asarray(self.prng.uniform(size=(self.n_hidden[i-1], h_units),
                                                    low=-0.01, high=0.01),
                                                    dtype=theano.config.floatX))

                param_idx += self.n_hidden[i-1] * h_units

        return param_idx, Wxf_init

    def initialize_output_gate_forward_weights(self, param_idx):
        self.Wxo = []
        Wxo_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):
            if i == 0:
                self.Wxo.append(self.theta[param_idx:(param_idx + self.n_in * \
                                          h_units)].reshape((self.n_in, h_units)))
                Wxo_init = np.append(Wxo_init,
                                                np.asarray(self.prng.uniform(size=(self.n_in, h_units),
                                                low=-0.01, high=0.01),
                                                dtype=theano.config.floatX))
                param_idx += self.n_in * self.n_hidden[0]

            else:
                self.Wxo.append(self.theta[param_idx:(param_idx + self.n_hidden[i-1] * \
                                          h_units)].reshape((self.n_hidden[i-1], h_units)))
        
                Wxo_init = np.append(Wxo_init,
                                        np.asarray(self.prng.uniform(size=(self.n_hidden[i-1], h_units),
                                                    low=-0.01, high=0.01),
                                                    dtype=theano.config.floatX))

                param_idx += self.n_hidden[i-1] * h_units

            self.Wxo[i].name = 'Wxo_{0}'.format(i)
        return param_idx, Wxo_init

    def initialize_output_weights(self, param_idx):
        self.W_out = self.theta[param_idx:(param_idx + self.n_hidden[-1] * \
                                           self.n_out)].reshape((self.n_hidden[-1], self.n_out))
        self.W_out.name = 'W_out'

        W_out_init = np.asarray(self.prng.uniform(size=(self.n_hidden[-1], self.n_out),
                                                  low=-0.01, high=0.01),
                                                  dtype=theano.config.floatX)
        param_idx += self.n_hidden[-1] * self.n_out
        return param_idx, W_out_init

    def initialize_hidden_states(self, param_idx):

        h0_param_idx = param_idx

        self.h0 = []
        h0_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.h0.append(self.theta[param_idx:(param_idx + h_units)])
            self.h0[i].name = 'h0_{0}'.format(i)
            h0_init = np.append(h0_init, np.zeros((h_units,), dtype=theano.config.floatX))
            param_idx += h_units

        self.h0_as_vector = self.theta[h0_param_idx:param_idx]
        self.h0_as_vector.name = 'h0_as_vector'

        c0_param_idx = param_idx

        self.c0 = []
        c0_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.c0.append(self.theta[param_idx:(param_idx + h_units)])
            self.c0[i].name = 'c0_{0}'.format(i)
            c0_init = np.append(h0_init, np.zeros((h_units,), dtype=theano.config.floatX))
            param_idx += h_units

        self.c0_as_vector = self.theta[c0_param_idx:param_idx]
        self.c0_as_vector.name = 'c0_as_vector'
        return param_idx, h0_init, c0_init

    def initialize_bias_input_gate(self, param_idx):

        self.bi = []
        bi_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.bi.append(self.theta[param_idx:(param_idx + h_units)])
            self.bi[i].name = 'bi_{0}'.format(i)
            bi_init = np.append(bi_init, self.bias_init[0]*np.ones((h_units,),
                                                                        dtype=theano.config.floatX))
            param_idx += h_units

        bi_init = np.array(bi_init)

        return param_idx, bi_init

    def initialize_bias_candidate_state(self, param_idx):

        self.bj = []
        bj_init = np.empty(0)
        for i, h_units in enumerate(self.n_hidden):

            self.bj.append(self.theta[param_idx:(param_idx + h_units)])
            self.bj[i].name = 'bj_{0}'.format(i)
            bj_init = np.append(bj_init, self.bias_init[1]*np.ones((h_units,),
                                                                        dtype=theano.config.floatX))
            param_idx += h_units

        bj_init = np.array(bj_init)
        return param_idx, bj_init

    def initialize_bias_forget_gate(self, param_idx):

        self.bf = []
        bf_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.bf.append(self.theta[param_idx:(param_idx + h_units)])
            self.bf[i].name = 'bf_{0}'.format(i)
            #FORGET TO 1!!!!!
            bf_init = np.append(bf_init, self.bias_init[2]*np.ones((h_units,),
                                                                        dtype=theano.config.floatX))
            param_idx += h_units

        bf_init = np.array(bf_init)
        return param_idx, bf_init

    def initialize_bias_output_gate(self, param_idx):
        self.bo = []
        bo_init = np.empty(0)

        for i, h_units in enumerate(self.n_hidden):

            self.bo.append(self.theta[param_idx:(param_idx + h_units)])
            self.bo[i].name = 'bo_{0}'.format(i)
            bo_init = np.append(bo_init, self.bias_init[3]*np.ones((h_units,),
                                                                        dtype=theano.config.floatX))
            param_idx += h_units

        bo_init = np.array(bo_init)
        return param_idx, bo_init

    def initialize_bias_output(self, param_idx) :
        self.by = self.theta[param_idx:(param_idx + self.n_out)]
        self.by.name = 'by'
        by_init = self.bias_init[4]*np.ones((self.n_out,), dtype=theano.config.floatX)
        param_idx += self.n_out
        return param_idx, by_init

    def pack_weights(self):
        """Packs weights and biases for convenience"""

        # for convenience
        self.params = []
        self.weights = []

        for w in self.Whi:
            self.params.append(w)
            self.weights.append(w)

        for w in self.Whj:
            self.params.append(w)
            self.weights.append(w)

        for w in self.Whf:
            self.params.append(w)
            self.weights.append(w)            

        for w in self.Who:
            self.params.append(w)
            self.weights.append(w)

        for w in self.Wxi:
            self.params.append(w)
            self.weights.append(w)

        for w in self.Wxj:
            self.params.append(w)
            self.weights.append(w)

        for w in self.Wxf:
            self.params.append(w)
            self.weights.append(w)            

        for w in self.Wxo:
            self.params.append(w)
            self.weights.append(w)            

        self.params.append(self.W_out)
        self.weights.append(w)            

        for h0 in self.h0:
            self.params.append(h0)

        for c0 in self.c0:
            self.params.append(c0)

        for bh in self.bi:
            self.params.append(bh)

        for bh in self.bj:
            self.params.append(bh)

        for bh in self.bf:
            self.params.append(bh)

        for bh in self.bo:
            self.params.append(bh)

        self.params.append(self.by)

    def define_network(self, input_variable):
        """Given the input variable of the network sets all variables and connections"""

        self.input = input_variable

        n_layers = len(self.n_hidden)

        # theta is a vector of all trainable parameters
        # it represents the value of W, W_in, W_out, h0, bh, by

        theta_shape = 4*np.sum(self.n_hidden ** 2) + 4*self.n_in * self.n_hidden[0] + \
                                    4*np.sum(self.n_hidden[:-1]*self.n_hidden[1:]) + self.n_hidden[-1] * self.n_out + \
                                    4*np.sum(self.n_hidden) + 2*np.sum(self.n_hidden) + self.n_out
        self.theta = theano.shared(value=np.zeros(theta_shape,
                                                  dtype=theano.config.floatX))

        self.initialize_theta(theta_shape)        


        self.theta_update = theano.shared(
            value=np.zeros(theta_shape, dtype=theano.config.floatX))

        # recurrent function (using tanh activation function) and arbitrary output
        # activation function
        def step(x_t, h_tm1, c_tm1):

            #h_t and c_t shape [batch_size, hidden_units]
            h_t = T.zeros_like(h_tm1)
            c_t = T.zeros_like(c_tm1)

            input_gate = T.zeros_like(h_tm1)
            candidate_gate = T.zeros_like(h_tm1)
            output_gate = T.zeros_like(h_tm1)
            forget_gate = T.zeros_like(h_tm1)

            idx = 0
            for i, h_units in enumerate(self.n_hidden):
                if i == 0:
                    input_gate = T.set_subtensor(input_gate[:,idx:(idx+h_units)],
                                self.activation[0](T.dot(x_t, self.Wxi[i]) +
                                T.dot(h_tm1[:,idx:(idx + h_units)], self.Whi[i]) + self.bi[i]))
                    candidate_gate = T.set_subtensor(candidate_gate[:,idx:(idx+h_units)],
                                self.activation[1](T.dot(x_t, self.Wxj[i]) + 
                                T.dot(h_tm1[:,idx:(idx + h_units)], self.Whj[i]) + self.bj[i]))
                    forget_gate = T.set_subtensor(forget_gate[:,idx:(idx+h_units)],
                                self.activation[2](T.dot(x_t, self.Wxf[i]) +
                                T.dot(h_tm1[:,idx:(idx + h_units)], self.Whf[i]) + self.bf[i]))
                    output_gate = T.set_subtensor(output_gate[:,idx:(idx+h_units)],
                                self.activation[3](T.dot(x_t, self.Wxo[i]) +
                                T.dot(h_tm1[:,idx:(idx + h_units)], self.Who[i]) + self.bo[i]))

                else:
                    input_gate = T.set_subtensor(input_gate[:,idx:(idx+h_units)],
                                self.activation[0](T.dot(h_t[:,(idx-self.n_hidden[i-1]):idx],
                                self.Wxi[i]) + T.dot(h_tm1[:,idx:(idx + h_units)], self.Whi[i]) +
                                                                                        self.bi[i]))
                    candidate_gate = T.set_subtensor(candidate_gate[:,idx:(idx+h_units)],
                                self.activation[1](T.dot(h_t[:,(idx-self.n_hidden[i-1]):idx],
                                self.Wxj[i]) + T.dot(h_tm1[:,idx:(idx + h_units)], self.Whj[i]) +
                                                                                        self.bj[i]))
                    forget_gate = T.set_subtensor(forget_gate[:,idx:(idx+h_units)],
                                self.activation[2](T.dot(h_t[:,(idx-self.n_hidden[i-1]):idx],
                                self.Wxf[i]) + T.dot(h_tm1[:,idx:(idx + h_units)], self.Whf[i]) +
                                                                                        self.bf[i]))
                    output_gate = T.set_subtensor(output_gate[:,idx:(idx+h_units)],
                                self.activation[3](T.dot(h_t[:,(idx-self.n_hidden[i-1]):idx],
                                self.Wxo[i]) + T.dot(h_tm1[:,idx:(idx + h_units)], self.Who[i]) +
                                                                                        self.bo[i]))

                c_t = T.set_subtensor(c_t[:,idx:(idx+h_units)],
                    c_tm1[:, idx:(idx+h_units)]*forget_gate[:, idx:(idx + h_units)] +
                    input_gate[:, idx:(idx + h_units)]*candidate_gate[:, idx:(idx + h_units)])                                                         #
                h_t = T.set_subtensor(h_t[:,idx:(idx+h_units)],
                    self.activation[4](c_t[:, idx:(idx+h_units)])*output_gate[:, idx:(idx+h_units)])

                idx += h_units

            y_t = T.dot(h_t[:,(idx-self.n_hidden[-1]):idx], self.W_out) + self.by
            return h_t, c_t, y_t


        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        # Note the implementation of weight-sharing h0 across variable-size
        # batches using T.ones multiplying h0
        # Alternatively, T.alloc approach is more robust


        [self.h, self.c, self.y_pred], _ = theano.scan(step,
                    sequences=self.input,
                    outputs_info=[T.alloc(self.h0_as_vector, self.input.shape[1],
                                          np.sum(self.n_hidden)), T.alloc(self.c0_as_vector, self.input.shape[1],
                                          np.sum(self.n_hidden)), None])
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        for w in self.weights:
            self.L1 += abs(w.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        for w in self.weights:
            self.L2_sqr += (w ** 2).sum()

        self.defined = True
