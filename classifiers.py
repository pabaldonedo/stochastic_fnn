import theano.tensor as T
import theano
import numpy as np
import logging
from types import IntType
from types import ListType
from types import FloatType
import json
import os
from optimizer import SGD
from optimizer import RMSProp
from optimizer import AdaDelta
from optimizer import AdaGrad
from optimizer import Adam
import mlp
from mlp import MLPLayer
from lbn import LBN
from LBNRNN import LBNRNN_module
from rnn import VanillaRNN
from rnn import LSTM
from util import flatten
from util import get_activation_function
from util import get_log_likelihood
from util import get_no_stochastic_log_likelihood
import warnings


class Classifier(object):
    """Builds a network that maps bones state to torque controls.
    For each bone (15 in total) an MLP is built. The output of the MLPs are concatenated.
    On top these MLPs there is a LBN/Noisy MLP network (depending on parameter noise_type).
    """

    def parse_inputs(self, n_in, n_out, mlp_n_in, mlp_n_hidden,
                     mlp_activation_names, lbn_n_hidden,
                     det_activations, stoch_activations, stoch_n_hidden,
                     log, likelihood_precision,
                     noise_type, batch_normalization):
        """Checks the type of the inputs and initializes instance variables.

        :type n_in: int.
        :param n_in: network input dimensionality.

        :type n_out: int.
        :param n_out: network output dimensionality.

        :type mlp_n_in: int.
        :param mlp_n_in: input dimensionality in bone MLPs.

        :type mlp_n_hidden: list of ints.
        :param mlp_n_hidden: dimensionalities of hidden layers in bone MLPs.

        :type mlp_activation_names: list of strings.
        :param mlp_activation_names: names of activation functions in bone MLPs.

        :type lbn_n_hidden: list of ints.
        :param lbn_n_hidden: dimensionalities of hidden layers in LBN network.

        :type det_activations: list of strings.
        :param det_activations: names of activation functions in deterministic part of
                                                                    LBN network layers.

        :type stoch_activations: list of strings.
        :param stoch_activations: names of activation functions in the MLP used in
                                           the tochastic part of LBN network layer.

        :type stoch_n_hidden: list of ints.
        :param stoch_activations: dimensionality of hidden layers in the MLP used in
                                  the stochastic part of LBN layer. Check LBN docs
                                  for more details.

        :type log: logging instance.
        :param log: log handler.

        :type likelihood_precision: int, float, double.
        :param likelihood_precision: precision parameter in log-likelihood function.
                                     Check util docs for more details.

        :type noise_type: string.
        :param noise_type: type of noise in the network. Can be "additive" or "multiplicative".
        """

        self.log = log
        if self.log is None:
            logging.basicConfig(level=logging.INFO)
            self.log = logging.getLogger()

        assert type(
            n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(mlp_n_in) is IntType, "mlp_n_in must be an integer: {0!r}".format(
            mlp_n_in)

        assert type(
            n_out) is IntType, "n_out must be an integer: {0!r}".format(n_out)
        assert type(mlp_n_hidden) is ListType, "mlp_n_hidden must be a list: {0!r}".format(
            mlp_n_hidden)
        assert type(
            mlp_activation_names) is ListType, "mlp_activation_names must be a list: {0!r}".format(
                mlp_n_hidden)
        assert type(lbn_n_hidden) is ListType, "lbn_n_hidden must be a list: {0!r}".format(
            lbn_n_hidden)
        assert type(det_activations) is ListType, "det_activations must be a list: {0!r}".format(
            det_activations)
        assert type(stoch_activations) is ListType, "stoch_activations must be a list: {0!r}". format(
            stoch_activations)
        assert type(stoch_n_hidden) is ListType, "stoch_n_hidden must be a list: {0!r}". format(
            stoch_n_hidden)
        assert type(batch_normalization) is bool, "batch_normalization must be bool. Given: {0!r}".format(
            batch_normalization)

        allowed_noise = ['multiplicative', 'additive']
        assert noise_type in allowed_noise, "noise_type must be one of {0!r}. Provided: {1!r}".format(
            allowed_noise, noise_type)

        self.batch_normalization = batch_normalization
        self.noise_type = noise_type
        self.lbn_n_hidden = lbn_n_hidden
        self.det_activations = det_activations
        self.stoch_activations = stoch_activations
        self.n_in = n_in
        self.stoch_n_hidden = stoch_n_hidden
        self.likelihood_precision = likelihood_precision
        self.n_out = n_out

    def set_up_mlp(self, mlp_n_hidden, mlp_activation_names, mlp_n_in, weights=None, timeseries_layer=False,
                   batch_normalization=False):
        """Defines the MLP networks for the 15 bones.

        :type mlp_n_hidden: list of ints.
        :param mlp_n_hidden: dimensionalities of hidden layers in bone MLPs.

        :type mlp_activation_names: list of strings.
        :param mlp_activation_names: names of activation functions in bone MLPs.

        :type mlp_n_in: int.
        :param mlp_n_in: input dimensionality in bone MLPs.
                         The first bone (hip bone) has mlp_n_in - 2 input dimensionality.

        :type weights: dict, None.
        :param weights: dictionary containing network information. Used for loading networks from file.
                        If None, random weights used for initialization.

        :type timeseries_layer: bool.
        :param timeseries: tells if the network is recurrent (True) or feedforward (False).
        """

        self.mlp_n_hidden = mlp_n_hidden
        self.bone_representations = [None] * 15
        self.mlp_activation_names = mlp_activation_names
        self.mlp_n_in = mlp_n_in
        for i in xrange(len(self.bone_representations)):
            if i == 0:
                bone_mlp = MLPLayer(mlp_n_in - 2, self.mlp_n_hidden, self.mlp_activation_names,
                                    input_var=self.x[:, :, :mlp_n_in - 2] if timeseries_layer
                                    else self.x[:, :mlp_n_in - 2],
                                    layers_info=None if weights is
                                    None else weights['bone_mlps'][i]['MLPLayer'],
                                    timeseries_network=timeseries_layer,
                                    batch_normalization=batch_normalization)
            else:
                bone_mlp = MLPLayer(mlp_n_in, self.mlp_n_hidden, self.mlp_activation_names,
                                    input_var=self.x[
                                        :, :, i * mlp_n_in - 2:(i + 1) * mlp_n_in - 2]
                                    if timeseries_layer else
                                    self.x[:, i * mlp_n_in -
                                           2:(i + 1) * mlp_n_in - 2],
                                    layers_info=None if weights is None else
                                    weights['bone_mlps'][i]['MLPLayer'],
                                    timeseries_network=timeseries_layer,
                                    batch_normalization=batch_normalization)
            self.bone_representations[i] = bone_mlp

    def __init__(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                 det_activations, stoch_activations, stoch_n_hidden=[-1], log=None, weights=None,
                 likelihood_precision=1, noise_type='multiplicative', batch_normalization=False):
        """
        :type n_in: int.
        :param n_in: network input dimensionality.

        :type n_out: int.
        :param n_out: network output dimensionality.

        :type mlp_n_in: int.
        :param mlp_n_in: input dimensionality in bone MLPs.

        :type mlp_n_hidden: list of ints.
        :param mlp_n_hidden: dimensionalities of hidden layers in bone MLPs.

        :type mlp_activation_names: list of strings.
        :param mlp_activation_names: names of activation functions in bone MLPs.

        :type lbn_n_hidden: list of ints.
        :param lbn_n_hidden: dimensionalities of hidden layers in LBN network.

        :type det_activations: list of strings.
        :param det_activations: names of activation functions in deterministic part of
                                                                    LBN network layers.

        :type stoch_activations: list of strings.
        :param stoch_activations: names of activation functions in the MLP used in
                                           the tochastic part of LBN network layer.

        :type stoch_n_hidden: list of ints.
        :param stoch_activations: dimensionality of hidden layers in the MLP used in
                                  the stochastic part of LBN layer. Check LBN docs
                                  for more details.

        :type log: logging instance.
        :param log: log handler.

        :type weights: dict, None.
        :param weights: dictionary containing network information. Used for loading networks from file.
                        If None, random weights used for initialization.

        :type likelihood_precision: int, float, double.
        :param likelihood_precision: precision parameter in log-likelihood function.
                                     Check util docs for more details.

        :type noise_type: string.
        :param noise_type: type of noise in the network. Can be "additive" or "multiplicative".
        """

        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.parse_inputs(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                          det_activations, stoch_activations, stoch_n_hidden, log, likelihood_precision,
                          noise_type, batch_normalization)

        self.set_up_mlp(mlp_n_hidden, mlp_activation_names, mlp_n_in,
                        weights=weights, batch_normalization=self.batch_normalization)

        self.lbn_input = T.concatenate([bone.output for bone in self.bone_representations] +
                                       [self.x[:, -2:]], axis=1)

        self.lbn = LBN(len(self.bone_representations) * self.mlp_n_hidden[-1] + 2, self.lbn_n_hidden,
                       self.n_out,
                       self.det_activations,
                       self.stoch_activations,
                       input_var=self.lbn_input,
                       layers_info=None if weights is None else
                       weights['lbn']['layers'],
                       likelihood_precision=self.likelihood_precision,
                       batch_normalization=self.batch_normalization)

        self.y = self.lbn.y
        self.m = self.lbn.m
        self.get_log_likelihood = theano.function(inputs=[self.x, self.lbn.y, self.lbn.m],
                                                  outputs=self.lbn.log_likelihood)

        mlp_params = [mlp_i.params for mlp_i in self.bone_representations]
        self.params = [mlp_params, self.lbn.params]
        if self.batch_normalization:
            self.frozen_weights = False
        self.predict = theano.function(
            inputs=[self.x, self.lbn.m], outputs=self.lbn.output)
        self.log.info("Network created with n_in: {0}, mlp_n_hidden: {1}, "
                      "mlp_activation_names: {2}, lbn_n_hidden: {3}, det_activations: {4}, "
                      "stoch_activations: {5}, n_out: {6}".format(
                          self.n_in, self.mlp_n_hidden, self.mlp_activation_names, self.lbn_n_hidden,
                          self.det_activations, self.stoch_activations, self.n_out))

    def freeze_weights(self, fname=None, dataset=None):
        assert fname is not None or dataset is not None, "with batch_normalization weights are required "\
            "to be frozen. For freezing a json file or a dataset is required"
        if fname is not None:
            if dataset is not None:
                warnings.warn(
                    "File name and dataset for freezing weights are provided. Only using the filename")
            with open(fname, 'r') as f:
                json.load  # TODO
        else:
            pass
            # TODO

    def predictTODO(self, fname=None, dataset=None):
        if self.batch_normalization:
            if not self.frozen_weights:
                self.freeze_weights(fname=fname, dataset=dataset)
                self.frozen_weights = True

    def save_network(self, fname):
        """Save network parameters in json format in fname"""

        output_string = self.generate_saving_string()
        with open(fname, 'w') as f:
            f.write(output_string)
        self.log.info("Network saved.")

    def generate_saving_string(self):
        """Generate json representation of network parameters"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                     "mlp_n_in": self.mlp_n_in, "mlp_n_hidden": self.mlp_n_hidden,
                                     "mlp_activation_names": self.mlp_activation_names,
                                     "lbn_n_hidden": self.lbn_n_hidden,
                                     "det_activations": self.det_activations,
                                     "stoch_activations": self.stoch_activations,
                                     "likelihood_precision": self.likelihood_precision,
                                     "noise_type": self.noise_type,
                                     "batch_normalization": self.batch_normalization})
        output_string += ",\"layers\": {\"bone_mlps\":["
        for i, bone in enumerate(self.bone_representations):
            if i > 0:
                output_string += ","
            output_string += "{\"MLPLayer\":"
            output_string += bone.generate_saving_string()
            output_string += "}"
        output_string += "]"
        output_string += ",\"lbn\":"
        output_string += self.lbn.generate_saving_string()
        output_string += "}}"

        return output_string

    def get_call_back(self, save_every, fname, epoch0, log_likelihood_constant=0, test_log_likelihood_constant=None):
        """Returns callback function to be sent to optimer for debugging and log purposes"""
        c = callBack(self, save_every, fname, epoch0,
                     log_likelihood_constant=log_likelihood_constant, test_log_likelihood_constant=test_log_likelihood_constant)
        return c.cback

    def get_cost(self):
        """Returns cost value to be optimized"""
        cost = -1. / self.x.shape[0] * self.lbn.log_likelihood
        return cost

    def fit(self, x, y, m, n_epochs, b_size, method, save_every=1, fname=None, epoch0=1,
            x_test=None, y_test=None, chunk_size=None, sample_axis=0):

        self.log.info("Number of training samples: {0}.".format(
            x.shape[sample_axis]))
        if x_test is not None:
            self.log.info("Number of test samples: {0}.".format(
                x_test.shape[sample_axis]))

        flat_params = flatten(self.params)
        cost = self.get_cost()
        compute_error = theano.function(inputs=[self.x, self.y], outputs=cost,
                                        givens={self.m: m})

        if sample_axis == 0:
            seq_length = 1
        else:
            seq_length = x.shape[0]

        log_likelihood_constant = x.shape[sample_axis] * seq_length * (np.log(
            m) + self.n_out * 0.5 * np.log(2 * np.pi * 1. / self.likelihood_precision))

        test_log_likelihood_constant = None
        if x_test is not None:

            test_log_likelihood_constant = x_test.shape[sample_axis] * seq_length * (np.log(
                m) + self.n_out * 0.5 * np.log(2 * np.pi * 1. / self.likelihood_precision))

        allowed_methods = ['SGD', "RMSProp", "AdaDelta", "AdaGrad", "Adam"]

        if method['type'] == allowed_methods[0]:
            opt = SGD(method['lr_decay_schedule'], method['lr_decay_parameters'],
                      method['momentum_type'], momentum=method['momentum'])
        elif method['type'] == allowed_methods[1]:
            opt = RMSProp(method['learning_rate'], method[
                          'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[2]:
            opt = AdaDelta(method['learning_rate'], method[
                           'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[3]:
            opt = AdaGrad(method['learning_rate'], method['epsilon'])
        elif method['type'] == allowed_methods[4]:
            opt = Adam(method['learning_rate'], method[
                       'b1'], method['b2'], method['e'])
        else:
            raise NotImplementedError("Optimization method not implemented. Choose one out of: {0}".format(
                allowed_methods))

        self.log.info("Fit starts with epochs: {0}, batch size: {1}, method: {2}".format(
            n_epochs, b_size, method))

        opt.fit(self.x, self.y, x, y, b_size, cost, flat_params, n_epochs,
                compute_error, self.get_call_back(
                    save_every, fname, epoch0, log_likelihood_constant=log_likelihood_constant,
                    test_log_likelihood_constant=test_log_likelihood_constant),
                extra_train_givens={self.m: m},
                x_test=x_test, y_test=y_test,
                chunk_size=chunk_size,
                sample_axis=sample_axis)

        """
        self.fiting_variables(b_size, train_set_x, train_set_y)

        gparams = [T.grad(cost, p) for p in flat_params]
        v = [theano.shared(value=np.zeros(
            th.shape.eval(), dtype=theano.config.floatX)) for th in flat_params]
        v_upds = [method['momentum'] * vi - method['learning_rate']
            * gp for vi, gp in zip(v, gparams)]
        upd = [(vi, v_updi) for vi, v_updi in zip(v, v_upds)]
        upd += [(p, p - method['learning_rate'] * gp + method['momentum'] * v_upd)
                 for p, gp, v_upd in zip(flat_params, gparams, v_upds)]

        train_model = theano.function(inputs=[self.index, self.n_ex],
                                    outputs=self.lbn.log_likelihood,
                                    updates=upd,
                                    givens={self.x: train_set_x[self.batch_start:self.batch_stop],
                                            self.lbn.y: train_set_y[self.batch_start:self.batch_stop],
                                            self.lbn.m: m})

        epoch = 0
        while epoch < n_epochs:
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_avg_cost = train_model(minibatch_idx, self.n_train)

            train_error = compute_error(x, y)
            log_likelihood = self.get_log_likelihood(x, y, m)
            self.log.info("epoch: {0} train_error: {1}, log_likelihood: {2} with".format(
                                                                    epoch + epoch0, train_error,
                                                                    log_likelihood))

        """

    def fiting_variables(self, batch_size, train_set_x, train_set_y, test_set_x=None):
        """DEPRECATED Sets useful variables for locating batches"""
        self.index = T.lscalar('index')    # index to a [mini]batch
        self.n_ex = T.lscalar('n_ex')      # total number of examples

        assert type(
            batch_size) is IntType or FloatType, "Batch size must be an integer."
        if type(batch_size) is FloatType:
            warnings.warn(
                'Provided batch_size is FloatType, value has been truncated')
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

    @classmethod
    def init_from_file(cls, fname, log=None):
        """Class method that loads network using information in json file fname

        :type fname: string.
        :param fname: filename (with path) containing network information.

        :type log: logging instance, None.
        :param log: logging instance to be used by the classifier.
        """
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['mlp_n_in'],
                                network_properties['mlp_n_hidden'],
                                network_properties['mlp_activation_names'],
                                network_properties['lbn_n_hidden'],
                                network_properties['det_activations'],
                                network_properties['stoch_activations'],
                                log=log,
                                weights=network_description['layers'],
                                likelihood_precision=network_properties[
                                    'likelihood_precision'],
                                noise_type=network_properties['noise_type'],
                                batch_normalization=False if 'batch_normalization'
                                not in network_properties.keys() else
                                network_properties['batch_normalization'])

        return loaded_classifier


class RecurrentClassifier(Classifier):
    """Builds a RNN on top of Classifier Network."""

    def parse_inputs(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                     lbn_n_out, det_activations, stoch_activations, stoch_n_hidden, likelihood_precision,
                     rnn_hidden, rnn_activations, rnn_type, log, noise_type):
        """Checks the type of the inputs and initializes instance variables.

        :type rnn_hidden: list of ints.
        :param rnn_hidden: dimensionality of hidden layers in the RNN.

        :type rnn_activations: list of strings.
        :param rnn_activations: names of activation functions of RNN.

        :type rnn_type: string.
        :param rnn_type: type of RNN (e.g. LSTM).
        """
        super(RecurrentClassifier, self).parse_inputs(n_in, n_out, mlp_n_in, mlp_n_hidden,
                                                      mlp_activation_names, lbn_n_hidden,
                                                      det_activations, stoch_activations,
                                                      stoch_n_hidden, log, likelihood_precision,
                                                      noise_type)

        assert type(
            lbn_n_out) is IntType, "lbn_n_out must be an integer: {0!r}".format(lbn_n_out)
        assert type(rnn_hidden) is ListType, "rnn_hidden must be a list: {0!r}".format(
            rnn_hidden)
        assert type(rnn_activations) is ListType, "rnn_activations must be a list: {0!r}".format(
            rnn_activations)
        self.lbn_n_out = lbn_n_out
        self.rnn_hidden = rnn_hidden
        self.rnn_activations = rnn_activations
        self.rnn_type = rnn_type

    def __init__(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                 lbn_n_out, det_activations, stoch_activations, likelihood_precision,
                 rnn_hidden, rnn_activations, rnn_type, stoch_n_hidden=[-1],
                 log=None, weights=None, noise_type="multiplicative"):
        """
        :type rnn_hidden: list of ints.
        :param rnn_hidden: dimensionality of hidden layers in the RNN.

        :type rnn_activations: list of strings.
        :param rnn_activations: names of activation functions of RNN.

        :type rnn_type: string.
        :param rnn_type: type of RNN (e.g. LSTM).
        """
        self.x = T.tensor3('x', dtype=theano.config.floatX)

        self.parse_inputs(n_in, n_out, mlp_n_in,
                          mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                          lbn_n_out, det_activations, stoch_activations, stoch_n_hidden, likelihood_precision,
                          rnn_hidden, rnn_activations, rnn_type,  log, noise_type)
        self.set_up_mlp(mlp_n_hidden, mlp_activation_names,
                        mlp_n_in, weights, timeseries_layer=True)

        self.lbn_input = T.concatenate([bone.output for bone in self.bone_representations] +
                                       [self.x[:, :, -2:]], axis=2)

        lbn_properties = {'n_in': len(self.bone_representations) * self.mlp_n_hidden[-1] + 2,
                          'n_hidden': self.lbn_n_hidden, 'n_out': lbn_n_out,
                          'det_activations': self.det_activations,
                          'stoch_activations': self.stoch_activations,
                          'stoch_n_hidden': self.stoch_n_hidden,
                          'input_var': self.lbn_input,
                          'layers': None if weights is None else weights['lbnrnn']['lbn']['layers']}

        rnn_properties = {'n_in': lbn_properties['n_out'],
                          'n_out': self.n_out,
                          'n_hidden': self.rnn_hidden,
                          'activations': self.rnn_activations,
                          'layers': None if weights is None else weights['lbnrnn']['rnn']['layers'],
                          'type': self.rnn_type}

        self.lbnrnn = LBNRNN_module(lbn_properties, rnn_properties, input_var=self.lbn_input, likelihood_precision=self.likelihood_precision,
                                    noise_type=noise_type)

        # Will be used for restarting the predictions
        self.rnn0 = []
        self.rnn0.append([l.h0.get_value(borrow=False)
                          for l in self.lbnrnn.rnn.hidden_layers])

        if self.rnn_type == 'LSTM':
            self.rnn0.append([l.c0.get_value(borrow=False)
                              for l in self.lbnrnn.rnn.hidden_layers])

        self.y = self.lbnrnn.y
        self.m = self.lbnrnn.lbn.m
        mlp_params = [mlp_i.params for mlp_i in self.bone_representations]
        self.params = [mlp_params, self.lbnrnn.params]
        self.get_log_likelihood = theano.function(inputs=[self.x, self.lbnrnn.y, self.lbnrnn.lbn.m],
                                                  outputs=self.lbnrnn.log_likelihood)

        self.output = self.lbnrnn.output
        self.predict_sequence = theano.function(
            inputs=[self.x, self.lbnrnn.lbn.m], outputs=self.output)

        # self.set_up_predict_one()
        self.log.info("Network created with n_in: {0}, mlp_n_hidden: {1}, "
                      "mlp_activation_names: {2}, lbn_n_hidden: {3}, det_activations: {4}, "
                      "stoch_activations: {5}, n_out: {6}".format(
                          self.n_in, self.mlp_n_hidden, self.mlp_activation_names, self.lbn_n_hidden,
                          self.det_activations, self.stoch_activations, self.n_out))

    def restart_prediction(self):
        for i, l in enumerate(self.lbnrnn.rnn.hidden_layers):
            l.h0.set_value(self.rnn0[0][i], borrow=False)
            if self.rnn_type == 'LSTM':
                l.c0.set_value(self.rnn0[1][i], borrow=False)

    def set_up_predict_one(self):
        warnings.warn("ONLY FOR ONE SAMPLE THE PREDICTION!!!")
        predict_upd = [(l.h0, l.output[-1].flatten())  # TODO multiple samples
                       for l in self.lbnrnn.rnn.hidden_layers]

        if self.rnn_type is "LSTM":
            predict_upd += [(l.c0, l.c_t[-1].flatten())  # TODO multiple samples
                            for l in self.lbnrnn.rnn.hidden_layers]

        self.predict_one = theano.function(
            inputs=[self.x, self.lbnrnn.lbn.m], outputs=self.output[-1], updates=predict_upd)

    def get_cost(self):
        """Returns cost value to be optimized"""
        cost = -1. / (self.x.shape[0] * self.x.shape[1]
                      ) * self.lbnrnn.log_likelihood

        return cost

    def generate_saving_string(self):
        """Generate json representation of network parameters"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                     "mlp_n_in": self.mlp_n_in, "mlp_n_hidden": self.mlp_n_hidden,
                                     "mlp_activation_names": self.mlp_activation_names,
                                     "lbn_n_hidden": self.lbn_n_hidden,
                                     "lbn_n_out": self.lbn_n_out,
                                     "det_activations": self.det_activations,
                                     "stoch_activations": self.stoch_activations,
                                     "rnn_hidden": self.rnn_hidden,
                                     "rnn_activations": self.rnn_activations,
                                     "likelihood_precision": self.likelihood_precision,
                                     "rnn_type": self.rnn_type,
                                     "noise_type": self.noise_type})
        output_string += ",\"layers\": {\"bone_mlps\":["
        for i, bone in enumerate(self.bone_representations):
            if i > 0:
                output_string += ","
            output_string += "{\"MLPLayer\":"
            output_string += bone.generate_saving_string()
            output_string += "}"
        output_string += "]"
        output_string += ",\"lbnrnn\":"
        output_string += self.lbnrnn.generate_saving_string()
        output_string += "}}"

        return output_string

    @classmethod
    def init_from_file(cls, fname, log=None):
        """Class method that loads network using information in json file fname

        :type fname: string.
        :param fname: filename (with path) containing network information.

        :type log: logging instance, None.
        :param log: logging instance to be used by the classifier.
        """
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['mlp_n_in'],
                                network_properties['mlp_n_hidden'],
                                network_properties['mlp_activation_names'],
                                network_properties['lbn_n_hidden'],
                                network_properties['lbn_n_out'],
                                network_properties['det_activations'],
                                network_properties['stoch_activations'],
                                network_properties['likelihood_precision'],
                                network_properties['rnn_hidden'],
                                network_properties['rnn_activations'],
                                network_properties['rnn_type'],
                                log=log,
                                weights=network_description['layers'],
                                noise_type=network_properties['noise_type'])

        return loaded_classifier


class RNNClassifier(object):
    """Builds a RNN on top of Classifier Network."""

    def parse_inputs(self, n_in, n_out, likelihood_precision, rnn_hidden, rnn_activations, rnn_type, log):
        """Checks the type of the inputs and initializes instance variables.

        :type rnn_hidden: list of ints.
        :param rnn_hidden: dimensionality of hidden layers in the RNN.

        :type rnn_activations: list of strings.
        :param rnn_activations: names of activation functions of RNN.

        :type rnn_type: string.
        :param rnn_type: type of RNN (e.g. LSTM).
        """
        assert type(rnn_hidden) is ListType, "rnn_hidden must be a list: {0!r}".format(
            rnn_hidden)
        assert type(rnn_activations) is ListType, "rnn_activations must be a list: {0!r}".format(
            rnn_activations)
        self.n_in = n_in
        self.n_out = n_out
        self.likelihood_precision = likelihood_precision
        self.rnn_hidden = rnn_hidden
        self.rnn_activations = rnn_activations
        self.rnn_type = rnn_type

    def __init__(self, n_in, n_out, likelihood_precision,
                 rnn_hidden, rnn_activations, rnn_type,
                 log=None, layers_info=None):
        """
        :type rnn_hidden: list of ints.
        :param rnn_hidden: dimensionality of hidden layers in the RNN.

        :type rnn_activations: list of strings.
        :param rnn_activations: names of activation functions of RNN.

        :type rnn_type: string.
        :param rnn_type: type of RNN (e.g. LSTM).
        """
        self.x = T.tensor3('x', dtype=theano.config.floatX)

        self.parse_inputs(n_in, n_out, likelihood_precision,
                          rnn_hidden, rnn_activations, rnn_type,  log)

        if self.rnn_type == 'rnn':
            self.rnn = VanillaRNN(self.n_in, self.rnn_hidden, self.n_out,
                                  self.rnn_activations,
                                  input_var=self.x,
                                  layers_info=None if layers_info is None else layers_info[
                                      'rnn']['layers'],
                                  stochastic_samples=False)
        elif self.rnn_type == 'LSTM':
            self.rnn = LSTM(self.n_in, self.rnn_hidden, self.n_out,
                            self.rnn_activations,
                            input_var=self.x,
                            layers_info=None if layers_info is None else layers_info[
                                'rnn']['layers'],
                            stochastic_samples=False)
        else:
            raise NotImplementedError

        self.y = T.tensor3('y', dtype=theano.config.floatX)

        self.params = []
        self.params.append(self.rnn.params)

        # Will be used for restarting the predictions
        self.rnn0 = []
        self.rnn0.append([l.h0.get_value(borrow=False)
                          for l in self.rnn.hidden_layers])

        if self.rnn_type == 'LSTM':
            self.rnn0.append([l.c0.get_value(borrow=False)
                              for l in self.rnn.hidden_layers])

        self.log = log
        if self.log is None:
            logging.basicConfig(level=logging.INFO)
            self.log = logging.getLogger()
        self.output = self.rnn.output
        self.predict_sequence = theano.function(
            inputs=[self.x], outputs=self.output)

        self.set_up_predict_one()
        self.log.info("Network created with n_in: {0}, n_out: {1}, rnn_hidden: {2}, rnn_activations: {3}".format(
            self.n_in, self.n_out, self.rnn_hidden, self.rnn_activations))

    def restart_prediction(self):
        for i, l in enumerate(self.rnn.hidden_layers):
            l.h0.set_value(self.rnn0[0][i], borrow=False)
            if self.rnn_type == 'LSTM':
                l.c0.set_value(self.rnn0[1][i], borrow=False)

    def set_up_predict_one(self):

        predict_upd = [(l.h0, l.output[-1].flatten())
                       for l in self.rnn.hidden_layers]

        if self.rnn_type is "LSTM":
            predict_upd += [(l.c0, l.c_t[-1].flatten())
                            for l in self.rnn.hidden_layers]

        self.predict_one = theano.function(
            inputs=[self.x], outputs=self.output[-1], updates=predict_upd)

    def fit(self, x, y, n_epochs, b_size, method, save_every=1, fname=None, epoch0=1, x_test=None,
            y_test=None, chunk_size=None, sample_axis=1):

        self.log.info("Number of training samples: {0}.".format(
            x.shape[sample_axis]))
        if x_test is not None:
            self.log.info("Number of test samples: {0}.".format(
                x_test.shape[sample_axis]))

        flat_params = flatten(self.params)
        cost = self.get_cost()
        compute_error = theano.function(inputs=[self.x, self.y], outputs=cost)

        if sample_axis == 0:
            seq_length = 1
        else:
            seq_length = x.shape[0]

        log_likelihood_constant = self.n_out * 0.5 * x.shape[sample_axis] * seq_length * np.log(2 * np.pi /
                                                                                                self.likelihood_precision)

        test_log_likelihood_constant = None
        if x_test is not None:
            test_log_likelihood_constant = self.n_out * 0.5 * x_test.shape[sample_axis] * seq_length * np.log(2 * np.pi /
                                                                                                              self.likelihood_precision)

        allowed_methods = ['SGD', "RMSProp", "AdaDelta", "AdaGrad", "Adam"]

        if method['type'] == allowed_methods[0]:
            opt = SGD(method['lr_decay_schedule'], method['lr_decay_parameters'],
                      method['momentum_type'], momentum=method['momentum'])
        elif method['type'] == allowed_methods[1]:
            opt = RMSProp(method['learning_rate'], method[
                          'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[2]:
            opt = AdaDelta(method['learning_rate'], method[
                           'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[3]:
            opt = AdaGrad(method['learning_rate'], method['epsilon'])
        elif method['type'] == allowed_methods[4]:
            opt = Adam(method['learning_rate'], method[
                       'b1'], method['b2'], method['e'])
        else:
            raise NotImplementedError(
                "Optimization method not implemented. Choose one out of: {0}".format(
                    allowed_methods))

        self.log.info("Fit starts with epochs: {0}, batch size: {1}, method: {2}".format(
            n_epochs, b_size, method))

        opt.fit(self.x, self.y, x, y, b_size, cost, flat_params, n_epochs,
                compute_error, self.get_call_back(save_every, fname, epoch0,
                                                  log_likelihood_constant=log_likelihood_constant,
                                                  test_log_likelihood_constant=test_log_likelihood_constant),
                x_test=x_test, y_test=y_test,
                chunk_size=chunk_size,
                sample_axis=1)

    def get_cost(self):
        """Returns cost value to be optimized"""
        cost = -1. / (self.x.shape[0] * self.x.shape[1]
                      ) * get_no_stochastic_log_likelihood(self.output, self.y,
                                                           self.likelihood_precision, True)

        return cost

    def get_call_back(self, save_every, fname, epoch0, log_likelihood_constant=0, test_log_likelihood_constant=None):
        """Returns callback function to be sent to optimer for debugging and log purposes"""
        c = callBack(self, save_every, fname, epoch0,
                     log_likelihood_constant=log_likelihood_constant, test_log_likelihood_constant=test_log_likelihood_constant)
        return c.cback

    def generate_saving_string(self):
        """Generate json representation of network parameters"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                     "rnn_hidden": self.rnn_hidden,
                                     "rnn_activations": self.rnn_activations,
                                     "likelihood_precision": self.likelihood_precision,
                                     "rnn_type": self.rnn_type})
        output_string += ",\"rnn\": "
        output_string += self.rnn.generate_saving_string()
        output_string += "}"

        return output_string

    def save_network(self, fname):
        """Save network parameters in json format in fname"""

        output_string = self.generate_saving_string()
        with open(fname, 'w') as f:
            f.write(output_string)
        self.log.info("Network saved.")

    @classmethod
    def init_from_file(cls, fname, log=None):
        """Class method that loads network using information in json file fname

        :type fname: string.
        :param fname: filename (with path) containing network information.

        :type log: logging instance, None.
        :param log: logging instance to be used by the classifier.
        """
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['likelihood_precision'],
                                network_properties['rnn_hidden'],
                                network_properties['rnn_activations'],
                                network_properties['rnn_type'],
                                log=log,
                                layers_info=network_description)

        return loaded_classifier


class callBack:
    """Call back class used for logging and debugging in the optimizer"""

    def __init__(self, classifier, save_every, fname, epoch0, log_likelihood_constant=0, test_log_likelihood_constant=None):
        """
        :type classifier: classifier instance.
        :param classifier: network being used.

        :type save_every: int.
        :param save_every: sets every how many epochs the network is saved.

        :type fname: string.
        :param fname: path and name of file where the network is saved.

        :type epoch0: int.
        :param epoch0: starting epoch number. If first time trained, set it to 1.

        :type log_likelihood_constant: float.
        :param log_likelihood_constant: constant to be subtracted from reported
        log_likelihood_constant.
        """

        self.epoch0 = epoch0
        self.train_log_likelihoods = []
        self.test_log_likelihoods = []
        self.epochs = []
        self.classifier = classifier
        self.log_likelihood_constant = log_likelihood_constant
        self.test_log_likelihood_constant = test_log_likelihood_constant

        opath = os.path.dirname(fname)
        file_name = os.path.basename(fname)
        like_file = '{0}/likelihoods/{1}.csv'.format(opath, file_name)

        self.likelihood_file(like_file)
        self.save_every = save_every

        network_name = '{0}/networks/{1}'.format(opath, file_name)
        if not os.path.exists('{0}/networks'.format(opath)):
            os.makedirs('{0}/networks'.format(opath))
        self.fname = network_name

    def likelihood_file(self, fname):
        """
        :type fname: string.
        :param fname: path and name of file where the likelihood is saved.
        """
        path_fname = os.path.dirname(fname)
        if not os.path.exists(path_fname):
            os.makedirs(path_fname)

        def save_likelihood(epochs, log_likelihoods, test_like=None):
            """
            :type epochs: numpy.array, list.
            :param epochs: array containing the epoch numbers to be saved.

            :type log_likelihood: numpy.array, list.
            :param log_likelihood: array containing the training log_likelihood to be saved.

            :type test_like: numpy.array, list.
            :param test_like: array containing the test log_likelihood to be saved.
            """
            with open(fname, 'a') as f:
                for e, l in zip(epochs, log_likelihoods):
                    f.write('{0},{1}\n'.format(e, l))
            if test_like is not None:
                test_fname = os.path.splitext(os.path.basename(fname))[0]

                with open('{0}/{1}_test.csv'.format(path_fname, test_fname), 'a') as f:
                    for e, l in zip(epochs, test_like):
                        f.write('{0},{1}\n'.format(e, l))
            self.classifier.log.info("Log likelihoods saved.")

        self.save_likelihood = save_likelihood

    def cback(self, epoch, n_samples, train_log_likelihood=None, opt_parameters=None,
              test_log_likelihood=None, n_test=None):
        """Call back function to be sent to optimizer.

        :type epoch: int.
        :param epoch: current epoch.

        :type n_samples: int.
        :param n_samples: number of samples used in this epoch.

        :type train_log_likelihood: float, double.
        :param train_log_likelihood: current log_likelihood of training set.

        :type opt_parameters: list.
        :param opt_parameters: list describing the optimization settings.

        :type test_log_likelihood: float, double.
        :param test_log_likelihood: current log_likelihood of test set.

        :type n_test: int.
        :param n_test: number of samples in test set.
        """

        train_log_likelihood -= self.log_likelihood_constant
        train_error = -train_log_likelihood * 1. / n_samples
        if test_log_likelihood is None:

            self.classifier.log.info("epoch: {0} train_error: {1}, log_likelihood: {2} with"
                                     "options: {3}.".format(epoch + self.epoch0, train_error,
                                                            train_log_likelihood, opt_parameters))

        else:
            assert self.test_log_likelihood_constant is not None
            test_log_likelihood -= self.test_log_likelihood_constant
            test_error = -test_log_likelihood / n_test
            self.classifier.log.info("epoch: {0} train_error: {1}, test_error: {2} "
                                     "log_likelihood: {3}, test_log_likelihood: {4}, "
                                     "with options: {5}.".format(
                                         epoch + self.epoch0, train_error,
                                         test_error,
                                         train_log_likelihood,
                                         test_log_likelihood,
                                         opt_parameters))
            self.test_log_likelihoods.append(test_log_likelihood)

        self.epochs.append(epoch + self.epoch0)
        self.train_log_likelihoods.append(train_log_likelihood)
        if (epoch + 1) % self.save_every == 0 and self.fname is not None:
            self.classifier.save_network(
                "{0}_epoch_{1}.json".format(self.fname, epoch + self.epoch0))
            self.save_likelihood(self.epochs, self.train_log_likelihoods,
                                 test_like=None if test_error is
                                 None else self.test_log_likelihoods)
            self.train_log_likelihoods = []
            self.epochs = []
            if test_error is not None:
                self.test_log_likelihoods = []


class MLPClassifier(object):

    def __init__(self, n_in, n_out, mlp_n_hidden, mlp_activation_names,
                 likelihood_precision=1, layers_info=None, log=None,
                 batch_normalization=False):

        self.n_in = n_in
        self.n_out = n_out
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_activation_names = mlp_activation_names
        self.log = log
        self.likelihood_precision = likelihood_precision
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype=theano.config.floatX)
        self.params = []
        self.batch_normalization = batch_normalization

        self.mlp = MLPLayer(self.n_in, self.mlp_n_hidden, self.mlp_activation_names,
                            timeseries_network=False,
                            input_var=self.x,
                            layers_info=None if layers_info is None else layers_info[
                                'mlp'],
                            batch_normalization=self.batch_normalization)

        linear_activation = get_activation_function('linear')

        self.params.append(self.mlp.params)
        self.output_layer = mlp.HiddenLayer(self.mlp.output, self.mlp.hidden_layers[-1].n_out,
                                            self.n_out, 'linear', linear_activation,
                                            W_values=None if layers_info is None else layers_info[
                                                'output_layer']['W'],
                                            b_values=None if layers_info is None else layers_info[
                                                'output_layer']['b'],
                                            timeseries_layer=None,
                                            batch_normalization=self.batch_normalization,
                                            gamma_values=None if layers_info is None or
                                            'gamma_values' not in layers_info['output_layer'].keys() else layers_info['output_layer']['gamma_values'],
                                            beta_values=None if layers_info is None or 'beta_values' not in layers_info[
                                                'output_layer'].keys() else layers_info['output_layer']['beta_values'],
                                            epsilon=1e-12 if layers_info is None or 'epsilon' not in layers_info[
                                                'output_layer'].keys() else layers_info['output_layer']['epsilon'],
                                            fixed_means=False)

        self.params.append(self.output_layer.params)
        self.log = log
        if self.log is None:
            logging.basicConfig(level=logging.INFO)
            self.log = logging.getLogger()

        self.output = self.output_layer.output
        self.predict = theano.function(inputs=[self.x], outputs=self.output)

    def get_call_back(self, save_every, fname, epoch0, log_likelihood_constant=0, test_log_likelihood_constant=None):
        """Returns callback function to be sent to optimer for debugging and log purposes"""
        c = callBack(self, save_every, fname, epoch0,
                     log_likelihood_constant=log_likelihood_constant, test_log_likelihood_constant=test_log_likelihood_constant)
        return c.cback

    def fit(self, x, y, n_epochs, b_size, method, save_every=1, fname=None, epoch0=1, x_test=None,
            y_test=None, chunk_size=None, sample_axis=0):

        self.log.info("Number of training samples: {0}.".format(
            x.shape[sample_axis]))
        if x_test is not None:
            self.log.info("Number of test samples: {0}.".format(
                x_test.shape[sample_axis]))

        flat_params = flatten(self.params)
        cost = self.get_cost()
        compute_error = theano.function(inputs=[self.x, self.y], outputs=cost)

        if sample_axis == 0:
            seq_length = 1
        else:
            seq_length = x.shape[0]

        log_likelihood_constant = x.shape[
            sample_axis] * seq_length * 0.5 * self.n_out * np.log(2 * np.pi / self.likelihood_precision)

        test_log_likelihood_constant = None
        if x_test is not None:
            test_log_likelihood_constant = x_test.shape[
                sample_axis] * seq_length * 0.5 * self.n_out * np.log(2 * np.pi / self.likelihood_precision)

        allowed_methods = ['SGD', "RMSProp", "AdaDelta", "AdaGrad", "Adam"]

        if method['type'] == allowed_methods[0]:
            opt = SGD(method['lr_decay_schedule'], method['lr_decay_parameters'],
                      method['momentum_type'], momentum=method['momentum'])
        elif method['type'] == allowed_methods[1]:
            opt = RMSProp(method['learning_rate'], method[
                          'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[2]:
            opt = AdaDelta(method['learning_rate'], method[
                           'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[3]:
            opt = AdaGrad(method['learning_rate'], method['epsilon'])
        elif method['type'] == allowed_methods[4]:
            opt = Adam(method['learning_rate'], method[
                       'b1'], method['b2'], method['e'])
        else:
            raise NotImplementedError(
                "Optimization method not implemented. Choose one out of: {0}".format(
                    allowed_methods))

        self.log.info("Fit starts with epochs: {0}, batch size: {1}, method: {2}".format(
            n_epochs, b_size, method))

        opt.fit(self.x, self.y, x, y, b_size, cost, flat_params, n_epochs,
                compute_error, self.get_call_back(save_every, fname, epoch0,
                                                  log_likelihood_constant=log_likelihood_constant,
                                                  test_log_likelihood_constant=test_log_likelihood_constant),
                x_test=x_test, y_test=y_test,
                chunk_size=chunk_size,
                sample_axis=sample_axis)

    def get_cost(self):
        """Returns cost value to be optimized"""
        cost = -1. / self.x.shape[0] * get_no_stochastic_log_likelihood(self.output, self.y,
                                                                        self.likelihood_precision, False)
        return cost

    def generate_saving_string(self):
        """Generate json representation of network parameters"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                     "mlp_n_hidden": self.mlp_n_hidden,
                                     "mlp_activation_names": self.mlp_activation_names,
                                     "likelihood_precision": self.likelihood_precision,
                                     "batch_normalization": self.batch_normalization})

        output_string += ",\"mlp\":"
        output_string += self.mlp.generate_saving_string()
        output_string += ",\"output_layer\":"
        buffer_dict = {"n_in": self.output_layer.n_in, "n_out": self.output_layer.n_out,
                       "activation": self.output_layer.activation_name,
                       "W": self.output_layer.W.get_value().tolist(),
                       "b": self.output_layer.b.get_value().tolist(),
                       "timeseries": self.output_layer.timeseries}

        if self.batch_normalization:
            buffer_dict[
                'gamma_values'] = self.output_layer.gamma.get_value().tolist()
            buffer_dict[
                'beta_values'] = self.output_layer.beta.get_value().tolist()
            buffer_dict['epsilon'] = self.output_layer.epsilon

        output_string += json.dumps(buffer_dict)

        output_string += "}"

        return output_string

    def save_network(self, fname):
        """Save network parameters in json format in fname"""

        output_string = self.generate_saving_string()
        with open(fname, 'w') as f:
            f.write(output_string)
        self.log.info("Network saved.")

    @classmethod
    def init_from_file(cls, fname, log=None):
        """Class method that loads network using information in json file fname

        :type fname: string.
        :param fname: filename (with path) containing network information.

        :type log: logging instance, None.
        :param log: logging instance to be used by the classifier.
        """
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['mlp_n_hidden'],
                                network_properties['mlp_activation_names'],
                                network_properties['likelihood_precision'],
                                log=log,
                                layers_info=network_description,
                                batch_normalization=False if 'batch_normalization' not in network_properties.keys() else network_properties['batch_normalization'])

        return loaded_classifier


class RecurrentMLP(object):

    def __init__(self, n_in, n_out, mlp_n_hidden, mlp_activation_names,
                 rnn_hidden, rnn_activations, rnn_type,
                 likelihood_precision=1, layers_info=None, log=None):

        self.n_in = n_in
        self.n_out = n_out
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_activation_names = mlp_activation_names
        self.rnn_hidden = rnn_hidden
        self.rnn_activations = rnn_activations
        self.rnn_type = rnn_type
        self.log = log
        self.likelihood_precision = likelihood_precision
        self.x = T.tensor3('x', dtype=theano.config.floatX)
        self.y = T.tensor3('y', dtype=theano.config.floatX)
        self.params = []

        self.mlp = MLPLayer(self.n_in, self.mlp_n_hidden, self.mlp_activation_names,
                            timeseries_network=True,
                            input_var=self.x,
                            layers_info=None if layers_info is None else layers_info['mlp'])

        self.params.append(self.mlp.params)

        rnn_types = ['rnn', 'LSTM']
        if self.rnn_type == rnn_types[0]:
            self.rnn = VanillaRNN(int(self.mlp.hidden_layers[-1].n_out), self.rnn_hidden,
                                  self.n_out, self.rnn_activations,
                                  layers_info=None if layers_info is None else layers_info[
                'rnn']['layers'],
                input_var=self.mlp.output,
                stochastic_samples=False)
        elif self.rnn_type == rnn_types[1]:
            self.rnn = LSTM(int(self.mlp.hidden_layers[-1].n_out), self.rnn_hidden, self.n_out,
                            self.rnn_activations, input_var=self.mlp.output,
                            layers_info=None if layers_info is None else layers_info[
                'rnn']['layers'],
                stochastic_samples=False)
        else:
            raise NotImplementedError
        self.params.append(self.rnn.params)

        self.log = log
        if self.log is None:
            logging.basicConfig(level=logging.INFO)
            self.log = logging.getLogger()

        self.output = self.rnn.output
        self.set_up_predict_one()
        self.predict = theano.function(inputs=[self.x], outputs=self.output)

    def restart_prediction(self):
        for i, l in enumerate(self.rnn.hidden_layers):
            l.h0.set_value(self.rnn0[0][i], borrow=False)
            if self.rnn_type == 'LSTM':
                l.c0.set_value(self.rnn0[1][i], borrow=False)

    def set_up_predict_one(self):
        warnings.warn("ONLY FOR ONE SAMPLE THE PREDICTION!!!")
        predict_upd = [(l.h0, l.output[-1].flatten())  # TODO multiple samples
                       for l in self.rnn.hidden_layers]

        if self.rnn_type is "LSTM":
            predict_upd += [(l.c0, l.c_t[-1].flatten())  # TODO multiple samples
                            for l in self.rnn.hidden_layers]

        self.predict_one = theano.function(
            inputs=[self.x], outputs=self.output[-1], updates=predict_upd)

    def get_call_back(self, save_every, fname, epoch0, log_likelihood_constant=0, test_log_likelihood_constant=None):
        """Returns callback function to be sent to optimer for debugging and log purposes"""
        c = callBack(self, save_every, fname, epoch0,
                     log_likelihood_constant=log_likelihood_constant, test_log_likelihood_constant=test_log_likelihood_constant)
        return c.cback

    def fit(self, x, y, n_epochs, b_size, method, save_every=1, fname=None, epoch0=1, x_test=None,
            y_test=None, chunk_size=None):

        self.log.info("Number of training samples: {0}.".format(
            x.shape[1]))
        if x_test is not None:
            self.log.info("Number of test samples: {0}.".format(
                x_test.shape[1]))

        flat_params = flatten(self.params)
        cost = self.get_cost()
        compute_error = theano.function(inputs=[self.x, self.y], outputs=cost)

        log_likelihood_constant = x.shape[
            0] * x.shape[1] * 0.5 * x.shape[2] * np.log(2 * np.pi * 1. / np.sqrt(self.likelihood_precision))

        test_log_likelihood_constant = None
        if x_test is not None:
            test_log_likelihood_constant = x_test.shape[
                0] * x_test.shape[1] * 0.5 * x_test.shape[2] * np.log(2 * np.pi * 1. / np.sqrt(self.likelihood_precision))

        allowed_methods = ['SGD', "RMSProp", "AdaDelta", "AdaGrad", "Adam"]

        if method['type'] == allowed_methods[0]:
            opt = SGD(method['lr_decay_schedule'], method['lr_decay_parameters'],
                      method['momentum_type'], momentum=method['momentum'])
        elif method['type'] == allowed_methods[1]:
            opt = RMSProp(method['learning_rate'], method[
                'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[2]:
            opt = AdaDelta(method['learning_rate'], method[
                'rho'], method['epsilon'])
        elif method['type'] == allowed_methods[3]:
            opt = AdaGrad(method['learning_rate'], method['epsilon'])
        elif method['type'] == allowed_methods[4]:
            opt = Adam(method['learning_rate'], method[
                'b1'], method['b2'], method['e'])
        else:
            raise NotImplementedError(
                "Optimization method not implemented. Choose one out of: {0}".format(
                    allowed_methods))

        self.log.info("Fit starts with epochs: {0}, batch size: {1}, method: {2}".format(
            n_epochs, b_size, method))

        opt.fit(self.x, self.y, x, y, b_size, cost, flat_params, n_epochs,
                compute_error, self.get_call_back(
                    save_every, fname, epoch0, log_likelihood_constant=log_likelihood_constant,
                    test_log_likelihood_constant=test_log_likelihood_constant),
                x_test=x_test, y_test=y_test,
                chunk_size=chunk_size,
                sample_axis=1)

    def get_cost(self):
        """Returns cost value to be optimized"""
        cost = -1. / (self.x.shape[0] * self.x.shape[1]) * get_no_stochastic_log_likelihood(
            self.output, self.y,
            self.likelihood_precision, True)
        return cost

    def generate_saving_string(self):
        """Generate json representation of network parameters"""
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                     "mlp_n_hidden": self.mlp_n_hidden,
                                     "mlp_activation_names": self.mlp_activation_names,
                                     "likelihood_precision": self.likelihood_precision,
                                     "rnn_hidden": self.rnn_hidden,
                                     "rnn_type": self.rnn_type,
                                     "rnn_activations": self.rnn_activations})

        output_string += ",\"mlp\":"
        output_string += self.mlp.generate_saving_string()
        output_string += ",\"rnn\":"
        output_string += self.rnn.generate_saving_string()
        output_string += "}"

        return output_string

    def save_network(self, fname):
        """Save network parameters in json format in fname"""

        output_string = self.generate_saving_string()
        with open(fname, 'w') as f:
            f.write(output_string)
        self.log.info("Network saved.")

    @classmethod
    def init_from_file(cls, fname, log=None):
        """Class method that loads network using information in json file fname

        :type fname: string.
        :param fname: filename (with path) containing network information.

        :type log: logging instance, None.
        :param log: logging instance to be used by the classifier.
        """
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['mlp_n_hidden'],
                                network_properties['mlp_activation_names'],
                                network_properties['rnn_hidden'],
                                network_properties['rnn_activations'],
                                network_properties['rnn_type'],
                                likelihood_precision=network_properties[
            'likelihood_precision'],
            log=log,
            layers_info=network_description)

        return loaded_classifier
