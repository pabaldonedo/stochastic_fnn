import theano
import theano.tensor as T
import numpy as np
from util import parse_activations


class OutputLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, V=None):
        self.input = input_var
        if V is None:
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

            V = theano.shared(value=W_values, name='V', borrow=True)
        self.n_in = n_in
        self.n_out = n_out
        self.W = V
        self.params = [self.W]
        self.activation = activation
        def h_step(x):
            a = T.dot(x, self.W.T)
            output = self.activation(a)
            return a, output
        self.output = self.input
        [self.a, self.output], _ = theano.scan(h_step, sequences=[self.input])


class DetHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, m=None, W=None, b=None):
        """
        Typical deterministic hidden layer: Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input_var: theano.tensor.dmatrix
        :param input_var: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input_var
        if W is None:
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

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.activation = activation

        def h_step(x):
            no_bias = T.dot(x, self.W.T)
            a = no_bias + self.b
            output = self.activation(a)
            return no_bias, a, output

        if m is None:
            [self.no_bias, self.a, self.output], _ = theano.scan(h_step, sequences=self.input)
        else:
            [self.no_bias, self.a, self.output], _ = theano.scan(h_step, non_sequences=self.input,
                                                                outputs_info=[None]*3, n_steps=m)
        # parameters of the model
        self.params = [self.W, self.b]


class StochHiddenLayer(object):

    def __init__(self, rng, trng, input_var, n_in, n_hidden, n_out, activations):
        """
        stochastic hidden layer:
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type trng: theano.tensor.shared_randomstreams.RandomStreams
        :param trng: a random number generator used to sample

        :type input_var: theano.tensor.dmatrix
        :param input_var: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        self.input = input_var
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.hidden_layers = [None]*(self.n_hidden.size+1)

        weights = [None]*(self.n_hidden.size+1)
        biases = [None]*(self.n_hidden.size+1)

        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.input, self.n_in, h,
                                                                                    activations[i])
            else:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.hidden_layers[i-1].output,
                                                            self.n_hidden[i-1], h, activations[i])
            weights[i] = self.hidden_layers[i].W
            biases[i] = self.hidden_layers[i].b

        self.hidden_layers[-1] = DetHiddenLayer(rng, self.hidden_layers[-2].output,
                                                    self.n_hidden[-1], self.n_out, activations[-1])
        weights[-1] = self.hidden_layers[-1].W
        biases[-1] = self.hidden_layers[-1].b

        self.params = weights + biases
        self.ph = self.hidden_layers[-1].output
        sample = trng.uniform(size=self.ph.shape)
        self.output = T.lt(sample, self.ph)

class LBNHiddenLayer():

    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation, det_activation_name,
                                stoch_n_hidden, stoch_activations, stoch_activation_names, m=None):
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.det_activation = det_activation_name
        self.stoch_activation = stoch_activation_names
        self.det_layer = DetHiddenLayer(rng, input_var, n_in, n_out, det_activation, m=m)

        #If -1, same hidden units
        stoch_n_hidden = np.array([i if i > -1 else n_out for i in stoch_n_hidden])
        self.stoch_layer = StochHiddenLayer(rng, trng, self.det_layer.no_bias,
                                                    n_out, stoch_n_hidden, n_out, stoch_activations)

        self.output = self.stoch_layer.output*self.det_layer.output
        self.params = self.det_layer.params + self.stoch_layer.params


class LBN:
    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden):

        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.trng = T.shared_randomstreams.RandomStreams(1234)
        self.rng = np.random.RandomState(0)
        self.m = T.iscalar('M') 

        self.parse_properties(n_in, n_hidden, n_out, det_activations, stoch_activations,
                                                                                    stoch_n_hidden)
        self.define_network()

    def define_network(self):

        self.hidden_layers = [None]*self.n_hidden.size
        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng, self.x, self.n_in,
                                        h, self.det_activation[i], self.det_activation_names[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_names, m=self.m)
            else:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng,
                                        self.hidden_layers[i-1].output,
                                        self.n_hidden[i-1], h, self.det_activation[i],
                                        self.det_activation_names[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_names)

            self.params += self.hidden_layers[i].params

        self.output_layer = OutputLayer(self.rng, self.hidden_layers[-1].output,
                                                    n_hidden[-1], n_out, self.det_activation[-1])

        self.params += self.output_layer.params
        self.output = self.output_layer.output

        self.predict = theano.function(inputs=[self.x, self.m], outputs=self.output)

    def parse_properties(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                                                                                stoch_n_hidden):
        self.n_hidden = np.array(n_hidden)
        self.n_out = n_out
        self.n_in = n_in
        self.stoch_n_hidden = [np.array(i) for i in stoch_n_hidden]
        self.det_activation_names = det_activations
        self.det_activation, self.det_activation_prime = parse_activations(det_activations)
        self.stoch_activation_names = stoch_activations 
        self.stoch_activation, self.stoch_activation_prime = parse_activations(stoch_activations)    


if __name__ == '__main__':
    n_in = 10
    n_hidden = [4, 5]
    n_out = 2
    det_activations = ['linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    stoch_n_hidden = [-1]
    m = 2
    n = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)
    print n.predict(np.random.randn(3,10), m).shape
