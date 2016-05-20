import types
from types import IntType
from types import ListType
from types import FloatType
import numpy as np
import theano
import theano.tensor as T
import logging
import json
from util import parse_activations
from rnn import VanillaRNN
from lbn import LBN
from util import get_log_likelihood


class LBNRNN_module(object):

    def __init__(self, lbn_properties, rnn_definition, likelihood_precision=1, input_var=None):
      
        self.lbn = LBN(lbn_properties['n_in'], lbn_properties['n_hidden'], lbn_properties['n_out'],
                                                    lbn_properties['det_activations'],
                                                    lbn_properties['stoch_activations'],
                                                    lbn_properties['stoch_n_hidden'],
                                                    timeseries_network=True,
                                                    layers_info=lbn_properties['layers']
                                                    if 'layers' in lbn_properties.keys() else None,
                                                    input_var=input_var,
                                                    likelihood_precision=likelihood_precision)

        if input_var is None:
            self.x = self.lbn.x
        
        else:
            self.x = input_var
        self.likelihood_precision = likelihood_precision
        self.y = T.tensor3('y', dtype=theano.config.floatX)
        self.n_in = lbn_properties['n_in']
        self.n_out = rnn_definition['n_out']

        self.rnn = VanillaRNN(self.lbn.n_out, rnn_definition['n_hidden'], self.n_out,
                                                    rnn_definition['activations'],
                                                    rng=self.lbn.rng,
                                                    input_var=self.lbn.output,
                                                    layers_info=rnn_definition['layers']
                                                    if 'layers' in rnn_definition.keys() else None)

        self.params = [self.lbn.params] + [self.rnn.params]
        self.output = self.rnn.output
        self.predict = theano.function(inputs=[self.x, self.lbn.m], outputs=self.lbn.output)
        self.log_likelihood = get_log_likelihood(self.output, self.y, self.likelihood_precision, True)
        
        self.get_log_likelihood = theano.function(inputs=[self.x, self.y, self.lbn.m],
                                                  outputs=self.log_likelihood)

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
        self.n_train = train_set_x.get_value(borrow=True).shape[1]
        self.n_train_batches = int(np.ceil(1.0 * self.n_train / batch_size))
        if test_set_x is not None:
            self.n_test = test_set_x.get_value(borrow=True).shape[1]
            self.n_test_batches = int(np.ceil(1.0 * self.n_test / batch_size))

    def lbn_pretrain(self, x, y, m, learning_rate, epochs, batch_size):
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

        flat_params = [p for layer in self.params[0]  for p in layer]
        gparams = [T.grad(-1./(x.shape[0]*x.shape[1])*self.lbn.log_likelihood, p) for p in flat_params]
        upd = [(param, param - learning_rate * gparam)
                for param, gparam in zip(flat_params, gparams)]
        self.pretrain_model = theano.function(inputs=[self.index, self.n_ex],
                                    outputs=[self.lbn.log_likelihood, self.lbn.debugger, self.lbn.output],
                                    updates=upd,
                                    givens={self.x: train_set_x[:, self.batch_start:self.batch_stop],
                                            self.lbn.y: train_set_y[:, self.batch_start:self.batch_stop],
                                            self.lbn.m: m})

        log_likelihood = []
        for e in xrange(1,epochs+1):
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_likelihood, d, o = self.pretrain_model(minibatch_idx, self.n_train)
                
            log_likelihood.append(self.get_log_likelihood(x,y,m))
            print "Epoch {0} log likelihood: {1}".format(e, log_likelihood[-1])
        return log_likelihood

    def fit(self, x, y, m, learning_rate, epochs, batch_size, fname=None):
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

        flat_params = [p for network in self.params for layer in network  for p in layer]
        gparams = [T.grad(-1./(x.shape[0]*x.shape[1])*self.log_likelihood, p) for p in flat_params]
        upd = [(param, param - learning_rate * gparam)
                for param, gparam in zip(flat_params, gparams)]
        self.train_model = theano.function(inputs=[self.index, self.n_ex],
                                    outputs=[self.log_likelihood, self.debugger, self.output],
                                    updates=upd,
                                    givens={self.x: train_set_x[:, self.batch_start:self.batch_stop],
                                            self.y: train_set_y[:, self.batch_start:self.batch_stop],
                                            self.lbn.m: m})

        log_likelihood = []
        for e in xrange(1,epochs+1):
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_likelihood, d, o = self.train_model(minibatch_idx, self.n_train)
            log_likelihood.append(self.get_log_likelihood(x,y,m))
            print "Epoch {0} log likelihood: {1}".format(e, log_likelihood[-1])

        if fname is not None:
            self.save_network(fname)
        return log_likelihood

    def generate_saving_string(self):
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in":self.n_in,
                "n_out":self.n_out})
        output_string += ",\"lbn\":"
        output_string += self.lbn.generate_saving_string()
        output_string += ",\"rnn\":"
        output_string += self.rnn.generate_saving_string()
        output_string += "}"
        return output_string
    def save_network(self, fname):
        output_string = self.generate_saving_string()
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

        network_properties = network_description['network_properties']
        lbn_definition = network_description['lbn']
        rnn_definition = network_description['rnn']

        loaded_lbn = cls(lbn_definition['network_properties'],
                        rnn_definition['network_properties'])

        return loaded_lbn


if __name__ == '__main__':
    from lbn import LBN
    import cPickle, gzip
    import numpy as np
    np.random.seed(0)
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    x_train = train_set[0]
    y_train = train_set[1]
    x_val = valid_set[0]
    y_val = valid_set[1]
    f.close()

    x_train = x_train[:20].reshape(2,10,-1)
    y_train = x_train[:, :,14*28:].copy()
    x_train = x_train[:, :,:14*28]
    batch_size = 20

    n_in = x_train.shape[2]
    n_hidden = [200, 100]
    n_out = y_train.shape[2]
    det_activations = ['linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    stoch_n_hidden = [-1]
    m = 5
    #n = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)
    #lbn = LBN.init_from_file("last_network.json")
    lbnrnn = LBNRNN_module({'n_in':n_in, 'n_hidden':n_hidden, 'n_out':n_out,
                            'det_activations':det_activations,
                            'stoch_activations':stoch_activations,
                            'stoch_n_hidden': stoch_n_hidden}, [20], 1, ['linear', 'linear', 'linear'])
    #y_hat = lbnrnn.predict(x_train, m)
    y = lbnrnn.predict(x_train,m)
    import ipdb
    ipdb.set_trace()
