import numpy as np
import scipy.stats as stats
import logging
from util import flatten
from lbn import LBN
from classifiers import callBack
from optimizer import SGD
from optimizer import RMSProp
from optimizer import AdaDelta
from optimizer import AdaGrad
from optimizer import Adam
import theano
import theano.tensor as T
import logging
from util import flatten
from types import IntType
from types import ListType
from types import FloatType
import os
import matplotlib.pyplot as plt


class MultimodalGenerator(object):

    def __init__(self, mu_extreme=np.array([30, 230]), std_extreme=7,
                        mu_intermediate=np.array([80, 150]), std_intermediate=7,
                        mu_perfect=115, std_perfect=10, p_extreme=np.array([0.5, 0.5]),
                        p_intermediate=np.array([0.5, 0.5])):

        self.mu_extreme = mu_extreme
        self.std_extreme = std_extreme
        self.p_extreme = p_extreme
        assert np.sum(self.p_extreme) == 1

        self.mu_intermediate = mu_intermediate
        self.std_intermediate = std_intermediate
        self.p_intermediate = p_intermediate
        assert np.sum(self.p_intermediate) == 1

        self.mu_perfect = mu_perfect
        self.std_perfect = std_perfect

    def get_true_distributions(self, x_axis):

        normal1 = stats.norm.pdf(x_axis, self.mu_extreme[0], self.std_extreme)*self.p_extreme[0] +\
                self.p_extreme[1]*stats.norm.pdf(x_axis, self.mu_extreme[1], self.std_extreme)

        normal2 = stats.norm.pdf(x_axis, self.mu_intermediate[0], self.std_intermediate)*self.p_intermediate[0] +\
                self.p_intermediate[1]*stats.norm.pdf(x_axis, self.mu_intermediate[1], self.std_intermediate)

        normal3 = stats.norm.pdf(x_axis, self.mu_perfect, self.std_perfect)

        return normal1, normal2, normal3


    def generate_classes(self, n=1000, n_classes=3):

        assert 0 < n_classes <=3
        x = np.zeros((n,1), dtype=theano.config.floatX)
        y = np.zeros((n,1), dtype=theano.config.floatX)
        for i in xrange(n):
            x[i] = np.random.randint(n_classes)
            y[i] = self.gen_sample(x[i])

        return x, y

    def gen_sample(self, c):

        if c == 0:
            mode = np.random.binomial(1,0.5) < 0.5
            return np.random.normal(loc=self.mu_extreme[0], scale=self.std_extreme)*mode +\
                        (1-mode)*np.random.normal(loc=self.mu_extreme[1], scale=self.std_extreme)

        if c == 1:
            mode = np.random.binomial(1,0.5) < 0.5
            return np.random.normal(loc=self.mu_intermediate[0], scale=self.std_intermediate)*mode\
            + (1-mode)*np.random.normal(loc=self.mu_intermediate[1], scale=self.std_intermediate)

        return np.random.normal(loc=self.mu_perfect, scale=self.std_perfect)


class Classifier(object):
    """Builds a network that maps bones state to torque controls.
    For each bone (15 in total) an MLP is built. The output of the MLPs are concatenated.
    On top these MLPs there is a LBN/Noisy MLP network (depending on parameter noise_type).
    """

    def parse_inputs(self, n_in, n_out, lbn_n_hidden,
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
    
        assert type(
            n_out) is IntType, "n_out must be an integer: {0!r}".format(n_out)
        
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

    def __init__(self, n_in, n_out, lbn_n_hidden,
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
        self.parse_inputs(n_in, n_out, lbn_n_hidden,
                          det_activations, stoch_activations, stoch_n_hidden, log, likelihood_precision,
                          noise_type, batch_normalization)

        self.lbn = LBN(self.n_in, self.lbn_n_hidden,
                       self.n_out,
                       self.det_activations,
                       self.stoch_activations,
                       input_var=self.x,
                       layers_info=None if weights is None else
                       weights['lbn']['layers'],
                       likelihood_precision=self.likelihood_precision,
                       batch_normalization=self.batch_normalization)

        self.y = self.lbn.y
        self.m = self.lbn.m
        self.get_log_likelihood = theano.function(inputs=[self.x, self.lbn.y, self.lbn.m],
                                                  outputs=self.lbn.log_likelihood)

        self.params = self.lbn.params
        if self.batch_normalization:
            self.frozen_weights = False
        self.predict = theano.function(
            inputs=[self.x, self.lbn.m], outputs=self.lbn.output)

        self.log.info("Network created with n_in: {0}, lbn_n_hidden: {1}, det_activations: {2}, "
                      "stoch_activations: {3}, n_out: {4}".format(
                          self.n_in, self.lbn_n_hidden,
                          self.det_activations, self.stoch_activations, self.n_out))

    
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
            x_test=None, y_test=None, chunk_size=None, sample_axis=0, batch_logger=None):

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
                sample_axis=sample_axis,
                batch_logger=batch_logger, model=self)


def check_predictor(cl, mgen, inputs, mu=0, std=1):

    n = 3000
    colors = ['b', 'r', 'g']
    for ci,c in enumerate(inputs):
        output = np.zeros((n,1))
        for i in xrange(n):
            output[i] = (cl.predict(np.asarray(np.array(c).reshape(1,1), dtype=theano.config.floatX),1)[0])*std+mu

        plt.hist(output, bins=100, normed=True, color=colors[ci])


    x_axis = np.linspace(0, 255, 100)
    normal1, normal2, normal3 = mgen.get_true_distributions(x_axis)
    plt.plot(x_axis, normal1, color=colors[0])
    plt.plot(x_axis, normal2, color=colors[1])
    plt.plot(x_axis, normal3, color=colors[2])
    plt.show()

def main():
    mgen = MultimodalGenerator()
    x, y = mgen.generate_classes(n_classes=3)

    muy = np.mean(y, axis=0)
    stdy = np.std(y, axis=0)

    y_transform = (y-muy)*1./stdy
    n_in = 1
    n_hidden = [10]
    n_out = 1
    det_activations = ['linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    cl = Classifier(n_in, n_out, n_hidden, det_activations, stoch_activations,likelihood_precision=1)

    cl.params[0][2].set_value(-2*np.ones(cl.params[0][2].get_value().shape, dtype=theano.config.floatX))
    cl.params[0][4].set_value(-2*np.ones(cl.params[0][4].get_value().shape, dtype=theano.config.floatX))

    idx = np.random.permutation(np.arange(x.shape[0]))
    x_train = x[idx[:800]]
    x_test = x[idx[800:]]
    y_train = y_transform[idx[:800]]
    y_test = y_transform[idx[800:]]

    b_size = 1
    opt_type = 'SGD'
    dropout = False
    lr = .1
    method = {'type': opt_type, 'lr_decay_schedule': 'constant',
              'lr_decay_parameters': [lr],
              'momentum_type': 'none', 'momentum': 0.01, 'b1': 0.9,
              'b2': 0.999, 'epsilon': 1e-8, 'rho': 0.95, 'e': 1e-8,
              'learning_rate': lr, 'dropout': dropout}
    n_epochs = 100
    chunk_size = None
    epoch0 = 0
    save_every = 1001
    m = 1
    opath = 'multimodal_trial'
    if not os.path.exists(opath):
        os.makedirs(opath)
    fname = "{0}/net".format(opath)

    cl.fit(x_train, y_train, m, n_epochs, b_size, method, save_every=save_every, fname=fname, epoch0=epoch0,
                x_test=None, y_test=None, chunk_size=None, sample_axis=0)


    #Best normal distribution:

    muy0 =np.mean(y_train[x_train==0], axis=0)
    muy1 =np.mean(y_train[x_train==1], axis=0)
    muy2 =np.mean(y_train[x_train==2], axis=0)

    muy_norm = np.array([muy0, muy1, muy2])

    norm_error = -1./x_train.shape[0] * (np.sum((y_train-muy_norm[x_train.astype(int)])**2)*-0.5 - np.log(2*np.pi))
    print norm_error

    check_predictor(cl, mgen, [0,1,2], mu=muy, std=stdy)
