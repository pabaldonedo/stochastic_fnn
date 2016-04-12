import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import types
from types import IntType
from types import ListType
from types import FloatType
import json
from util import parse_activations


class OutputLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name, V_values=None):
        """
        LBN output layer.
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

        :type V: numpy.array.
        :param V: initialization values of the weights.
        """
        self.input = input_var
        if V_values is None:
            V_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_out, n_in)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                V_values *= 4

        V = theano.shared(value=V_values, name='V', borrow=True)
        self.n_in = n_in
        self.n_out = n_out
        self.W = V
        self.params = [self.W]
        self.activation = activation
        self.activation_name = activation_name

        def h_step(x):
            a = T.dot(x, self.W.T)
            output = self.activation(a)
            return a, output
        [self.a, self.output], _ = theano.scan(h_step, sequences=[self.input])
        

class DetHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name,
                                            m=None, W_values=None, b_values=None, no_bias=False):
        """
        Deterministic hidden layer: Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out,).
        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type input_var: theano.tensor.dmatrix.
        :param input_var: a symbolic tensor of shape (n_samples, n_in) or (m, n_samples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.

        :type activation_name: string
        :param activation_name: name of activation function.

        :type m: int.
        :param m: number of samples to be drawn in the layer and the input is consider constant.
                If b is None input is treat as a scan input sequence.

        :type W_values: numpy.array.
        :param W_values: initialization values of the weights.

        :type b_values: numpy array.
        :param b_values: initialization values of the bias.

        :type no_bias: bool.
        :param no_bias: sets if the layer has bias variable or not.
        """
        self.input = input_var
        self.no_bias = no_bias
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        self.m = m
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

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b_values is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        if no_bias is False:
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        if no_bias is False:
            self.b = b
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        self.activation = activation

        def h_step(x):
            no_bias_output = T.dot(x, self.W.T)
            if self.no_bias:
                a = no_bias_output
            else:
                a = no_bias_output + self.b
            output = self.activation(a)
            return no_bias_output, a, output

        if m is None:
            [self.no_bias_output, self.a, self.output], _ = theano.scan(
                                                                h_step, sequences=self.input)
        else:
            [self.no_bias_output, self.a, self.output], _ = theano.scan(
                                                                h_step, non_sequences=self.input,
                                                                outputs_info=[None]*3, n_steps=m)


class StochHiddenLayer(object):
    """
    Stochastic hidden MLP that are included in each LBN hidden layer.
    """
    def __init__(self, rng, trng, input_var, n_in, n_hidden, n_out, activations, activation_names,
                                                                                    mlp_info=None):
        """
        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.
        
        :type trng: theano.tensor.shared_randomstreams.RandomStreams.
        :param trng: a random number generator used to sample.

        :type input_var: theano.tensor.dmatrix.
        :param input_var: a symbolic tensor of shape (n_examples, n_in).

        :type n_in: int.
        :param n_in: input dimensionality.

        :type n_hidden: list of ints.
        :param n_hidden: list that defines the hidden units of the MLP.

        :type n_out: int.
        :param n_out: number of hidden units.

        :type activation: theano.Op or function.
        :param activation: Non linearity to be applied in the hidden layer.
        
        :type activation_names: list of strings.
        :param activation_names: list containing the name of the activation functions in the MLP.

        :type mlp_info: dict.
        :param mlp_info: dictionary containing the information of the mlp as generated in
                        LBN.save_network().
        """

        self.input = input_var
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.activation_names = activation_names
        self.hidden_layers = [None]*(self.n_hidden.size+1)
        self.params = [None]*(self.n_hidden.size+1)*2


        #Builds hidden layers of the MLP.
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.input, self.n_in, h,
                                                            activations[i], activation_names[i],
                                                            W_values=None if mlp_info is None else
                                                            np.array(mlp_info[i]['detLayer']['W']),
                                                            b_values=None if mlp_info is None else
                                                            np.array(mlp_info[i]['detLayer']['b']))
            else:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.hidden_layers[i-1].output,
                                                            self.n_hidden[i-1], h,
                                                            activations[i], activation_names[i],
                                                            W_values=None if mlp_info is None else
                                                            np.array(mlp_info[i]['detLayer']['W']),
                                                            b_values=None if mlp_info is None else
                                                            np.array(mlp_info[i]['detLayer']['b']))
            self.params[2*i] = self.hidden_layers[i].W
            self.params[2*i+1] = self.hidden_layers[i].b

        #Output layer of MLP.
        self.hidden_layers[-1] = DetHiddenLayer(rng, self.hidden_layers[-2].output,
                                                    self.n_hidden[-1], self.n_out,
                                                    activations[-1], activation_names[-1],
                                                    W_values=None if mlp_info is None else
                                                            np.array(mlp_info[-1]['detLayer']['W']),
                                                    b_values=None if mlp_info is None else
                                                            np.array(mlp_info[-1]['detLayer']['b']))
        self.params[-2] = self.hidden_layers[-1].W
        self.params[-1] = self.hidden_layers[-1].b

        #Sample from a Bernoulli distribution in each unit with a probability equal to the MLP
        #ouput.
        self.ph = self.hidden_layers[-1].output
        sample = trng.uniform(size=self.ph.shape)

        #Gradient that will be used is the one defined as "G3" in "Techniques for Learning Binary
        #stochastic feedforward Neural Networks" by Tapani Raiko, Mathias Berglund, Guillaum Alain
        #and Laurent Dinh. For this we need to propagate the gradient in the stochastic units
        #through ph. For this reason we use disconnected_grad() in epsilon.
        epsilon = theano.gradient.disconnected_grad(T.lt(sample, self.ph) - self.ph)
        self.output = self.ph + epsilon


class LBNHiddenLayer():
    """
    Layer of the LBN. It is made of a deterministic layer and a stochastic layer.
    """
    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation,
                                stoch_n_hidden, stoch_activations,
                                det_activation_name=None, stoch_activation_names=None, m=None,
                                det_W=None, det_b=None, stoch_mlp_info=None):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights.

        :type trng: theano.Tensor.shared_randomstreams.RandomStreams.
        :param trng: a random number generator used for drawing samples.

        :type input_var: theano.Tensor or theano.sahred_variable.
        :param input_var: input variable of the layer.

        :type n_in: int.
        :param n_in: number of input units of the layer.

        :type n_out: int.
        :param n_out: number of output units of the layer.

        :type det_activation: theano.Op or function.
        :param det_activation: Non linearity to be applied in the deterministic layer.

        :type stoch_n_hidden: list of ints.
        :param stoch_n_hidden: list that defines the hidden units of the stochastic MLP.

        :type stoch_activations: list of theano.Op or functions.
        :param stoch_activations: list of activation function for the stochastic MLP.

        :type: det_activation_name: string.
        :param stoch_activations_names: name of the deterministic activation.

        :type: stoch_activation_names: list of strings.
        :param stoch_activations_names: list of the names of the stochastic activations in the MLP.

        :type: m: int.
        :param m: number of samples to be drawn in the layer.

        :type det_W: numpy.array.
        :param det_W: initialization values of the weights in the deterministic layer.

        :type det_b: numpy.array.
        :param det_b: initialization values of the biases in the deterministic layer.

        :type stoch_mlp_info: dict.
        :param stoch_mlp_info: dictionary containing the information of the mlp as generated in
                            LBN.save_network().
        """

        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.det_activation = det_activation_name
        self.stoch_activation = stoch_activation_names
        self.m = m
        self.det_layer = DetHiddenLayer(rng, input_var, n_in, n_out, det_activation,
                                        det_activation_name, m=m, no_bias=True, 
                                        W_values=det_W, b_values=det_b)
           
        #If -1, same hidden units
        stoch_n_hidden = np.array([i if i > -1 else n_out for i in stoch_n_hidden])
        self.stoch_layer = StochHiddenLayer(rng, trng, self.det_layer.no_bias_output,
                                                    n_out, stoch_n_hidden, n_out,
                                                    stoch_activations, stoch_activation_names,
                                                    mlp_info=stoch_mlp_info)

        self.output = self.stoch_layer.output*self.det_layer.output
        self.params = self.det_layer.params + self.stoch_layer.params


class LBN:
    """
    Linearizing Belief Net (LBN) as explained in "Predicting Distributions with
    Linearizing Belief Networks" paper by Yann N. Dauphin and David Grangie from Facebook AI
    Research.
    """
    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                                                        stoch_n_hidden=[-1], keep_undefined=False):
        """
        :type n_in: int.
        :param n_in: input dimensionality of the network.

        :type n_hidden: list of ints.
        :param n_hidden: list that defines the dimensionality of the hidden layers.

        :type n_out: int.
        :param n_out: output dimensionality of the network.

        :type det_activations: list of strings.
        :param det_activations: list defining the activation function of the deterministic layers.
                                In the LBN paper all these activations are set to linear.

        :type stoch_activations: list of strings.
        :param stoch_activations: list defining the activations for the stochastic MLP. This
                                activation is the same in all layers of the LBN.

        :type stoch_n_hidden: list of ints.
        :param stoch_n_hidden: list that defines the hidden units of the stochastic MLP. Length of
                                stoch_n_hidden = length of stoch_activations - 1. If set to [-1] 
                                number of hidden units in the MLP are the same to the input
                                dimensionality.

        :type keep_undefined: bool.
        :param keep_undefined: used when loading network from file. Does not create the graph,
                            just the general network definition variables.
        """
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.trng = T.shared_randomstreams.RandomStreams(1234)
        self.rng = np.random.RandomState(0)
        self.m = T.lscalar('M') 
        assert type(n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(n_hidden)
        assert type(n_out) is IntType, "n_out must be an integer: {0!r}".format(n_out)
        assert type(det_activations) is ListType, "det_activations must be a list: {0!r}".format(
                                                                                    det_activations)
        assert len(n_hidden) == len(det_activations) - 1, "len(n_hidden) must be =="\
        " len(det_activations) - 1. n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden, 
                                                                                  det_activations)
        assert type(stoch_activations) is ListType, "stoch_activations must be a list: {0!r}".\
                                                                         format(stoch_activations)
        assert type(stoch_n_hidden) is ListType, "stoch_n_hidden must be a list: {0!r}".format(
                                                                                    stoch_n_hidden)
        assert stoch_n_hidden == [-1] or len(stoch_n_hidden) == len(stoch_activation) - 1, \
                "len(stoch_n_hidden) must be len(stoch_activations) -1 or stoch_n_hidden = [-1]."\
                " stoch_n_hidden = {0!r} and stoch_activations = {1!r}".format(stoch_n_hidden,
                                                                                stoch_activations)
        
        self.parse_properties(n_in, n_hidden, n_out, det_activations, stoch_activations,
                                                                                    stoch_n_hidden)
        if not keep_undefined:
            self.define_network()

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

    def define_network(self, layers_info=None):
        """
        Builds Theano graph of the network.
        """
        self.hidden_layers = [None]*self.n_hidden.size

        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng, self.x, self.n_in,
                                        h, self.det_activation[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        det_activation_name=self.det_activation_names[i],
                                        stoch_activation_names=self.stoch_activation_names,
                                        m=self.m,
                                        det_W=None if layers_info is None else
                                        np.array(
                                        layers_info['hidden_layers'][i]['LBNlayer']['detLayer']\
                                                                                            ['W']),
                                        det_b=None if layers_info is None else
                                        np.array(layers_info['hidden_layers'][i]\
                                                                    ['LBNlayer']['detLayer']['b']),
                                        stoch_mlp_info=None if layers_info is None else
                                        layers_info['hidden_layers'][i]['LBNlayer']['stochLayer'])
            else:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng,
                                        self.hidden_layers[i-1].output,
                                        self.n_hidden[i-1], h, self.det_activation[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        det_activation_name=self.det_activation_names[i],
                                        stoch_activation_names=self.stoch_activation_names, 
                                        det_W=None if layers_info is None else
                                        np.array(layers_info['hidden_layers'][i]['LBNlayer']\
                                                                                ['detLayer']['W']),
                                        det_b=None if layers_info is None else
                                        np.array(layers_info['hidden_layers'][i]['LBNlayer']\
                                                                                ['detLayer']['b']),
                                        stoch_mlp_info=None if layers_info is None else
                                        layers_info['hidden_layers'][i]['LBNlayer']['stochLayer'])

            self.params.append(self.hidden_layers[i].params)

        self.output_layer = OutputLayer(self.rng, self.hidden_layers[-1].output, n_hidden[-1], 
                                                            n_out, self.det_activation[-1],
                                                            self.det_activation_names[-1],
                                                            V_values=None 
                                                            if layers_info is None else np.array(
                                                            layers_info['output_layer']['W']))

        self.params.append(self.output_layer.params)
        self.output = self.output_layer.output
        exp_value = -0.5*T.sum((self.output - self.y.dimshuffle('x',0,1))**2, axis=2)
        max_exp_value = theano.ifelse.ifelse(T.lt(T.max(exp_value), -1*T.min(exp_value)),
                                                                T.max(exp_value), T.min(exp_value))
 
        self.log_likelihood = T.sum(T.log(T.sum(T.exp(exp_value - max_exp_value), axis=0)) +
                                                                                    max_exp_value)-\
                                self.y.shape[0]*(T.log(self.m)+self.y.shape[1]/2.*T.log(2*np.pi))

        self.predict = theano.function(inputs=[self.x, self.m], outputs=self.output)

    def fiting_variables(self, batch_size, train_set_x, train_set_y, test_set_x=None):
        """Sets useful variables for locating batches"""    
        self.index = T.lscalar('index')    # index to a [mini]batch
        self.n_ex = T.lscalar('n_ex')      # total number of examples

        assert type(batch_size) is IntType or FloatType, "Batch size must be an integer."
        if type(batch_size) is FloatType:
            warnings.warn('Provided batch_size is FloatType, value has been truncated')
            batch_size = int(batch_size)
        # Proper implementation of variable-batch size evaluation
        # Note that the last batch may be a smaller size
        # So we keep around the effective_batch_size (whose last element may
        # be smaller than the rest)
        # And weight the reported error by the batch_size when we average
        # Also, by keeping batch_start and batch_stop as symbolic variables,
        # we make the theano function easier to read
        self.batch_start = self.index * batch_size
        self.batch_stop = T.minimum(self.n_ex, (self.index + 1) * batch_size)
        self.effective_batch_size = self.batch_stop - self.batch_start

        self.get_batch_size = theano.function(inputs=[self.index, self.n_ex],
                                          outputs=self.effective_batch_size)

        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        self.n_train = train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches = int(np.ceil(1.0 * self.n_train / batch_size))
        if test_set_x is not None:
            self.n_test = test_set_x.get_value(borrow=True).shape[0]
            self.n_test_batches = int(np.ceil(1.0 * self.n_test / batch_size))


    def fit(self, x, y, m, learning_rate, epochs, batch_size):
        """
        :type x: numpy.array.
        :param x: input data of shape (n_samples, dimensionality).

        :type y: numpy.array.
        :param y: output data of shape (n_samples, 1).

        :type m: int.
        :param m: number of samples drawn from the network.

        :type learning_rate: float.
        :param learning_rate: step size for the stochastic gradient descent learning algoriithm.

        :type epochs: int.
        :param epochs: number of training epochs.

        :type batch_size: int
        :param batch_size: minibatch size for the SGD update.
        """

        train_set_x = theano.shared(np.asarray(x,
                                            dtype=theano.config.floatX))

        train_set_y = theano.shared(np.asarray(y,
                                            dtype=theano.config.floatX))


        self.fiting_variables(batch_size, train_set_x, train_set_y)

        flat_params = [p for layer in self.params for p in layer]
        gparams = [T.grad(-1./x.shape[0]*self.log_likelihood, p) for p in flat_params]

        upd = [(param, param - learning_rate * gparam)
                for param, gparam in zip(flat_params, gparams)]
        self.train_model = theano.function(inputs=[self.index, self.n_ex],
                                    outputs=self.log_likelihood,
                                    updates=upd,
                                    givens={self.x: train_set_x[self.batch_start:self.batch_stop],
                                            self.y: train_set_y[self.batch_start:self.batch_stop],
                                            self.m: m})

        self.get_log_likelihood = theano.function(inputs=[self.x, self.y, self.m],
                                                outputs=self.log_likelihood)
        log_likelihood = []
        for e in xrange(1,epochs+1):
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_likelihood = self.train_model(minibatch_idx, self.n_train)
            log_likelihood.append(self.get_log_likelihood(x,y,m))
            print "Epoch {0} log likelihood: {1}".format(e, log_likelihood[-1])
        self.save_network("last_network.json")

        plt.plot(np.arange(epochs),np.array(log_likelihood))
        plt.show()

    def save_network(self, fname):
        """
        Saves network to json file.

        :type fname: string.
        :param fname: file name (with local or global path) where to store the network.
        """
        output_string = "{\"network_properties\":"

        output_string += json.dumps({"n_in":self.n_in, "n_hidden":self.n_hidden.tolist(), "n_out":self.n_out,
                "det_activations":self.det_activation_names,
                "stoch_activations":self.stoch_activation_names, "stoch_n_hidden":[sh.tolist() for sh in self.stoch_n_hidden]})
        output_string += ",\"layers\":{\"hidden_layers\":["
        for k, l in enumerate(self.hidden_layers):
            det = l.det_layer
            stoch = l.stoch_layer
            if k > 0:
                output_string += ","
            output_string += "{\"LBNlayer\":{\"detLayer\":"
            output_string += json.dumps({"n_in":det.n_in,"n_out":det.n_out,
                    "activation":det.activation_name, "W":det.W.get_value().tolist(),
                    "b":det.b.get_value().tolist()if det.no_bias is False else None, "no_bias":det.no_bias})
            output_string += ", \"stochLayer\":"
            output_string += "["
            for i, hs in enumerate(stoch.hidden_layers):
                if i > 0:
                    output_string += ","
                output_string += "{\"detLayer\":"
                output_string += json.dumps({"n_in":hs.n_in, "n_out": hs.n_out,
                                    "activation":hs.activation_name, "W":hs.W.get_value().tolist(),
                                    "b": hs.b.get_value().tolist() if hs.no_bias is False else None,
                                    "no_bias": hs.no_bias})
                output_string += "}"
            output_string += "]}}"
        output_string += "]"
        output_string += ",\"output_layer\":"
        output_string += json.dumps({"n_in": self.output_layer.n_in,
                                    "n_out":self.output_layer.n_out,
                                    "activation":self.output_layer.activation_name,
                                    "W":self.output_layer.W.get_value().tolist()})
        output_string += "}}"
        with open('{0}'.format(fname), 'w') as f:
            f.write(output_string)

    @classmethod
    def init_from_file(cls, fname):
        """
        Loads a saved network from file fname.
        :type fname: string.
        :param fname: file name (with local or global path) from where to load the network.
        """
        with open(fname) as f:
            network_description = json.load(f)

        network_properties= network_description['network_properties']
        loaded_lbn = cls(network_properties['n_in'], network_properties['n_hidden'],
                        network_properties['n_out'], network_properties['det_activations'],
                        network_properties['stoch_activations'],
                        network_properties['stoch_n_hidden'], keep_undefined=True)

        loaded_lbn.define_network(network_description['layers'])
        return loaded_lbn

if __name__ == '__main__':

    import cPickle, gzip, numpy
    np.random.seed(0)
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    x_train = train_set[0]
    y_train = train_set[1]
    y_val = valid_set[1]
    f.close()

    x_train = x_train[:500]
    y_train = x_train[:, :14*28].copy()
    x_train = x_train[:, :14*28]
    batch_size = 100

    n_in = x_train.shape[1]
    n_hidden = [200]
    n_out = y_train.shape[1]
    det_activations = ['linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    stoch_n_hidden = [-1]
    m = 5
    n = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)

    epochs = 50
    lr = 0.01
    n.fit(x_train,y_train,m, lr,epochs, batch_size)
    y_hat = n.predict(x_train[0].reshape(1,-1), m)
    y_points = np.linspace(-1,10).reshape(1,-1,1)

    #distribution = np.sum(np.exp(-0.5*np.sum((y_points-y_hat)**2, axis=2)), axis=0)*1./(m*np.sqrt((2*np.pi)**y_hat.shape[2]))
    #plt.plot(y_points[0,:,0], distribution)
    #plt.show()

