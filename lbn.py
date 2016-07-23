import theano
import theano.tensor as T
import numpy as np
import logging
import os.path
from types import IntType
from types import ListType
from types import FloatType
import json
from util import parse_activations
from util import get_log_likelihood
from util import get_weight_init_values
from util import get_bias_init_values
from util import init_bn


class LBNOutputLayer(object):

    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name, V_values=None,
                 b_values=None,
                 timeseries_layer=False,
                 batch_normalization=False, gamma_values=None, beta_values=None, epsilon=1e-12,
                 fixed_means=False, stdb=None, mub=None, no_bias=True):
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

        :type V_values: numpy.array.
        :param V_values: initialization values of the weights.
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
        V_values = get_weight_init_values(
            n_in, n_out, activation=activation, rng=rng, W_values=V_values)

        b_values = get_bias_init_values(n_out, b_values=b_values)

        if not no_bias:
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        V = theano.shared(value=V_values, name='V', borrow=True)

        self.n_in = n_in
        self.n_out = n_out
        self.W = V
        self.params = [self.W]
        self.no_bias = no_bias

        if not self.no_bias:
            self.params += [self.b]
        self.activation = activation
        self.activation_name = activation_name
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

        if timeseries_layer:
            self.timeseries_layer()
        else:
            self.feedforward_layer()

    def timeseries_layer(self):
        def t_step(x):
            def h_step(x):

                a = T.dot(x, self.W.T)
                if not self.no_bias:
                    a += self.b
                return a

            a, _ = theano.scan(h_step, sequences=x)

            return a

        self.a, _ = theano.scan(t_step, sequences=self.input)

        if self.batch_normalization:
            if self.fixed_means:
                mub = self.mub
                stdb = self.stdb
            else:
                mub = T.sum(self.a, axis=(0, 1, 2)) * 1. / \
                    T.prod(self.a.shape[:-1])

                sigma2b = T.sum((self.a - mub)**2, axis=(0, 1, 2)) * \
                    1. / T.prod(self.a.shape[:-1])

                stdb = T.sqrt(sigma2b + self.epsilon)

            self.a = theano.tensor.nnet.bn.batch_normalization(
                self.a, self.gamma, self.beta, mub, stdb)

        self.output = self.activation(self.a)

    def feedforward_layer(self):
        def h_step(x):
            a = T.dot(x, self.W.T)
            if not self.no_bias:
                a += self.b
            return a

        self.a, _ = theano.scan(h_step, sequences=self.input)

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


class DetHiddenLayer(object):

    def __init__(self, rng, input_var, n_in, n_out, activation, activation_name, m=None,
                 W_values=None, b_values=None, no_bias=False, timeseries_layer=False,
                 batch_normalization=False, gamma_values=None, beta_values=None, epsilon=1e-12,
                 fixed_means=False, stdb=None, mub=None):
        """
        Deterministic hidden layer: Weight matrix W is of shape (n_out,n_in)
        and the bias vector b is of shape (n_out,).
        :type rng: numpy.random.RandomState.
        :param rng: a random number generator used to initialize weights.

        :type input_var: theano.tensor.dmatrix, theano.tensor.tensor3 or theano.tensor.tensor4.
        :param input_var: a symbolic tensor of shape (n_samples, n_in) or (m, n_samples, n_in).
                            If timeseries_layer=False then (t_points, n_samples, n_in) or
                            (t_points, m, n_samples, n_in).

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
        self.no_bias = no_bias
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation_name
        self.m = m
        W_values = get_weight_init_values(
            n_in, n_out, activation=activation, rng=rng, W_values=W_values)

        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = get_bias_init_values(n_out, b_values=b_values)

        if no_bias is False:
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        if no_bias is False:
            self.b = b
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        self.activation = activation
        self.timeseries = timeseries_layer

        self.batch_normalization = batch_normalization

        if self.batch_normalization:
            init_bn(self, n_out, gamma_values=gamma_values,
                    beta_values=beta_values)
            self.fixed_means = fixed_means
            self.params.append(self.gamma)
            self.params.append(self.beta)
            self.mub = None
            self.stdb = None
            self.epsilon = epsilon

        if timeseries_layer:
            self.timeseries_layer()
        else:
            self.feedforward_layer()

    def timeseries_layer(self):

        def t_step(x):
            def h_step(x):
                no_bias_output = T.dot(x, self.W.T)
                if self.no_bias:
                    a = no_bias_output
                else:
                    a = no_bias_output + self.b
                return no_bias_output, a

            if self.m is None:
                [no_bias_output, a], _ = theano.scan(
                    h_step, sequences=x)
            else:
                [no_bias_output, a], _ = theano.scan(h_step, non_sequences=x,
                                                     outputs_info=[
                                                         None] * 2,
                                                     n_steps=self.m)
            return no_bias_output, a

        [self.no_bias_output, self.a], _ = theano.scan(
            t_step, sequences=self.input)

        if self.batch_normalization:
            if self.fixed_means:
                mub = self.mub
                stdb = self.stdb

            else:
                mub = T.sum(self.a, axis=(0, 1, 2)) * 1. / \
                    T.prod(self.a.shape[:-1])

                sigma2b = T.sum((self.a - mub)**2, axis=(0, 1, 2)) * \
                    1. / T.prod(self.a.shape[:-1])

                stdb = T.sqrt(sigma2b + self.epsilon)

            self.a = theano.tensor.nnet.bn.batch_normalization(
                self.a, self.gamma, self.beta, mub, stdb)

        self.output = self.activation(self.a)

    def feedforward_layer(self):
        def h_step(x):
            no_bias_output = T.dot(x, self.W.T)
            if self.no_bias:
                a = no_bias_output
            else:
                a = no_bias_output + self.b
            return no_bias_output, a

        if self.m is None:
            [self.no_bias_output, self.a], _ = theano.scan(
                h_step, sequences=self.input)
        else:
            [self.no_bias_output, self.a], _ = theano.scan(
                h_step, non_sequences=self.input,
                outputs_info=[None] * 2,
                n_steps=self.m)

        if self.batch_normalization:
            mub = T.sum(self.a, axis=(0, 1)) * 1. / \
                T.prod(self.a.shape[:-1])

            sigma2b = T.sum((self.a - mub)**2, axis=(0, 1)) * \
                1. / T.prod(self.a.shape[:-1])

            sigmab = T.sqrt(sigma2b + self.epsilon)

            self.a = theano.tensor.nnet.bn.batch_normalization(
                self.a, self.gamma, self.beta, mub, sigmab)

        self.output = self.activation(self.a)


class StochHiddenLayerInterface(object):
    """
    Stochastic hidden MLP that are included in each LBN hidden layer.
    """

    def __init__(self, rng, trng, input_var, n_in, n_hidden, n_out, activations, activation_names,
                 mlp_info=None,
                 timeseries_layer=False,
                 batch_normalization=False, batch_normalization_info=None):
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
        self.hidden_layers = [None] * (self.n_hidden.size + 1)
        self.params = []  # [None] * (self.n_hidden.size + 1) * 2
        self.timeseries_layer = timeseries_layer
        self.rng = rng
        self.trng = trng
        self.batch_normalization = batch_normalization

        # Builds hidden layers of the MLP.
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.input, self.n_in, h,
                                                       activations[
                                                           i], activation_names[i],
                                                       W_values=None if mlp_info is None else
                                                       np.array(
                                                           mlp_info[i]['detLayer']['W']),
                                                       b_values=None if mlp_info is None else
                                                       np.array(
                                                           mlp_info[i]['detLayer']['b']),
                                                       timeseries_layer=self.timeseries_layer,
                                                       batch_normalization=self.batch_normalization,
                                                       gamma_values=None if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['gamma_values'],
                                                       beta_values=None if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['beta_values'],
                                                       epsilon=1e-12 if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['epsilon'],
                                                       fixed_means=False)
            else:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.hidden_layers[i - 1].output,
                                                       self.n_hidden[i - 1], h,
                                                       activations[
                                                           i], activation_names[i],
                                                       W_values=None if mlp_info is None else
                                                       np.array(
                                                           mlp_info[i]['detLayer']['W']),
                                                       b_values=None if mlp_info is None else
                                                       np.array(
                                                           mlp_info[i]['detLayer']['b']),
                                                       timeseries_layer=self.timeseries_layer,
                                                       batch_normalization=self.batch_normalization,
                                                       gamma_values=None if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['gamma_values'],
                                                       beta_values=None if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['beta_values'],
                                                       epsilon=1e-12 if batch_normalization_info is None else batch_normalization_info[i][
                                                           'detLayer']['epsilon'],
                                                       fixed_means=False)
            self.params += self.hidden_layers[i].params
            # self.params[2 * i] = self.hidden_layers[i].W
            # self.params[2 * i + 1] = self.hidden_layers[i].b

        # Output layer of MLP.
        self.hidden_layers[-1] = DetHiddenLayer(rng, self.hidden_layers[-2].output,
                                                self.n_hidden[-1], self.n_out,
                                                activations[-1], activation_names[-1],
                                                W_values=None if mlp_info is None else
                                                np.array(
                                                    mlp_info[-1]['detLayer']['W']),
                                                b_values=None if mlp_info is None else
                                                np.array(
                                                    mlp_info[-1]['detLayer']['b']),
                                                timeseries_layer=self.timeseries_layer,
                                                batch_normalization=self.batch_normalization,
                                                gamma_values=None if batch_normalization_info is None else batch_normalization_info[-1][
            'detLayer']['gamma_values'],
            beta_values=None if batch_normalization_info is None else batch_normalization_info[-1][
            'detLayer']['beta_values'],
            epsilon=1e-12 if batch_normalization_info is None else batch_normalization_info[-1][
            'detLayer']['epsilon'],
            fixed_means=False)
        # self.params[-2] = self.hidden_layers[-1].W
        # self.params[-1] = self.hidden_layers[-1].b
        self.params += self.hidden_layers[-1].params
        self.sample()

        def sample(self):
            pass


class StochHiddenLayerBernoulli(StochHiddenLayerInterface):

    def sample(self):
        # Sample from a Bernoulli distribution in each unit with a probability equal to the MLP
        # ouput.
        self.ph = self.hidden_layers[-1].output
        sample = self.trng.uniform(size=self.ph.shape)

        # Gradient that will be used is the one defined as "G3" in "Techniques for Learning Binary
        # stochastic feedforward Neural Networks" by Tapani Raiko, Mathias Berglund, Guillaum Alain
        # and Laurent Dinh. For this we need to propagate the gradient in the stochastic units
        # through ph. For this reason we use disconnected_grad() in epsilon.
        epsilon = theano.gradient.disconnected_grad(
            T.lt(sample, self.ph) - self.ph)
        self.output = self.ph + epsilon


class StochHiddenLayerGaussian(StochHiddenLayerInterface):

    def sample(self):

        self.ph = self.hidden_layers[-1].output
        sample = self.trng.normal(std=1., size=self.ph.shape)
        epsilon = theano.gradient.disconnected_grad(sample)
        self.output = self.ph + epsilon


class HiddenLayerInterface(object):
    """
    Layer of the LBN. It is made of a deterministic layer and a stochastic layer.
    """

    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation,
                 stoch_n_hidden, stoch_activations,
                 det_activation_name=None, stoch_activation_names=None, m=None,
                 det_W=None, det_b=None, stoch_mlp_info=None,
                 timeseries_layer=False,
                 batch_normalization=False, batch_normalization_info=None):
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
        self.timeseries_layer = timeseries_layer
        self.batch_normalization = batch_normalization

        self.det_layer = DetHiddenLayer(rng, input_var, n_in, n_out, det_activation,
                                        det_activation_name, m=m, no_bias=self.no_bias,
                                        W_values=det_W, b_values=det_b,
                                        timeseries_layer=self.timeseries_layer,
                                        batch_normalization=self.batch_normalization,
                                        gamma_values=None if batch_normalization_info is None else batch_normalization_info[
                                            'detLayer']['gamma_values'],
                                        beta_values=None if batch_normalization_info is None else batch_normalization_info[
                                            'detLayer']['beta_values'],
                                        epsilon=1e-12 if batch_normalization_info is None else batch_normalization_info[
                                            'detLayer']['epsilon'],
                                        fixed_means=False)

        # If -1, same hidden units
        stoch_n_hidden = np.array(
            [i if i > -1 else n_out for i in stoch_n_hidden])

        self.stoch_layer = self.stoch_hidden_layer_type(rng, trng, self.det_layer.no_bias_output,
                                                        n_out, stoch_n_hidden, n_out,
                                                        stoch_activations, stoch_activation_names,
                                                        mlp_info=stoch_mlp_info,
                                                        timeseries_layer=self.timeseries_layer,
                                                        batch_normalization=self.batch_normalization,
                                                        batch_normalization_info=None if batch_normalization_info is None else
                                                        batch_normalization_info['stochLayer'])
        # self.output = self.stoch_layer.output*self.det_layer.output
        self.define_output()
        self.params = self.det_layer.params + self.stoch_layer.params


class LBNHiddenLayer(HiddenLayerInterface):

    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation,
                 stoch_n_hidden, stoch_activations,
                 det_activation_name=None, stoch_activation_names=None, m=None,
                 det_W=None, det_b=None, stoch_mlp_info=None,
                 timeseries_layer=False,
                 batch_normalization=False, batch_normalization_info=None):

        self.stoch_hidden_layer_type = StochHiddenLayerBernoulli
        self.no_bias = True
        super(LBNHiddenLayer, self).__init__(rng, trng, input_var, n_in, n_out, det_activation,
                                             stoch_n_hidden, stoch_activations,
                                             det_activation_name=det_activation_name,
                                             stoch_activation_names=stoch_activation_names,
                                             m=m,
                                             det_W=det_W, det_b=det_b,
                                             stoch_mlp_info=stoch_mlp_info,
                                             timeseries_layer=timeseries_layer,
                                             batch_normalization=batch_normalization,
                                             batch_normalization_info=batch_normalization_info)

    def define_output(self):

        self.output = self.stoch_layer.output * self.det_layer.output


class NoisyMLPHiddenLayer(HiddenLayerInterface):

    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation,
                 stoch_n_hidden, stoch_activations,
                 det_activation_name=None, stoch_activation_names=None, m=None,
                 det_W=None, det_b=None, stoch_mlp_info=None,
                 timeseries_layer=False,
                 batch_normalization=False, batch_normalization_info=None):

        self.stoch_hidden_layer_type = StochHiddenLayerGaussian
        self.no_bias = False
        super(NoisyMLPHiddenLayer, self).__init__(rng, trng, input_var, n_in, n_out, det_activation,
                                                  stoch_n_hidden, stoch_activations,
                                                  det_activation_name=det_activation_name,
                                                  stoch_activation_names=stoch_activation_names,
                                                  m=m,
                                                  det_W=det_W, det_b=det_b,
                                                  stoch_mlp_info=stoch_mlp_info,
                                                  timeseries_layer=timeseries_layer,
                                                  batch_normalization=batch_normalization,
                                                  batch_normalization_info=batch_normalization_info)

    def define_output(self):
        self.output = self.stoch_layer.output + self.det_layer.output


class StochasticInterface(object):
    """
    Linearizing Belief Net (LBN) as explained in "Predicting Distributions with
    Linearizing Belief Networks" paper by Yann N. Dauphin and David Grangie from Facebook AI
    Research.
    """

    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                 stoch_n_hidden=[-1],
                 layers_info=None,
                 timeseries_network=False,
                 log=None,
                 input_var=None,
                 likelihood_precision=1.,
                 batch_normalization=False,
                 with_output_layer=True,
                 m=None):
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

        :type layers_info: dict or None.
        :param layers_info: used when loading network from file.

        :type precision: float or int.
        :param precision: in the Gaussian Mixture Model the value of precison diagonal matrix:
                         tau= precisoin * diag()
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
            if timeseries_network:
                self.y = T.tensor3('y', dtype=theano.config.floatX)
            else:
                self.y = T.matrix('y', dtype=theano.config.floatX)

        self.timeseries_network = timeseries_network
        self.trng = T.shared_randomstreams.RandomStreams()
        self.rng = np.random.RandomState()
        if m is None:
            self.m = T.lscalar('M')
        else:
            self.m = m
        self.log = log
        self.with_output_layer = with_output_layer

        if self.log is None:
            self.log = logging.getLogger()
        self.parse_properties(n_in, n_hidden, n_out, det_activations, stoch_activations,
                              stoch_n_hidden, likelihood_precision, batch_normalization)
        self.log.info('LBN Network created with n_in: {0}, n_hidden: {1}, n_out: {2}, '
                      'det_activations: {3}, stoch_activations: {4}, stoch_n_hidden: {5}'.format(
                          self.n_in, self.n_hidden, self.n_out, self.det_activation_names,
                          self.stoch_activation_names, self.stoch_n_hidden))

        self.define_network(layers_info=layers_info)
        self.predict = theano.function(
            inputs=[self.x, self.m], outputs=self.output)
        self.log_likelihood = get_log_likelihood(
            self.output, self.y, self.likelihood_precision, self.timeseries_network)
        self.regulizer_L2 = T.zeros(1)
        self.regulizer_L1 = T.zeros(1)
        for l in self.params:
            for p in l:
                self.regulizer_L2 += (p**2).sum()
                self.regulizer_L1 += p.sum()

        self.log.info('LBN Network defined.')

    def parse_properties(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                         stoch_n_hidden, likelihood_precision, batch_normalization):

        assert type(
            n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(n_hidden) is ListType, "n_hidden must be a list: {0!r}".format(
            n_hidden)
        assert type(
            n_out) is IntType, "n_out must be an integer: {0!r}".format(n_out)
        assert type(det_activations) is ListType, "det_activations must be a list: {0!r}".format(
            det_activations)

        if self.with_output_layer:
            assert len(n_hidden) == len(det_activations) - 1, "len(n_hidden) must be =="\
                " len(det_activations) - 1. n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden,
                                                                                               det_activations)
        else:
            assert len(n_hidden) == len(det_activations), "len(n_hidden) must be =="\
                " len(det_activations). n_hidden: {0!r} and det_activations: {1!r}".format(n_hidden,
                                                                                           det_activations)

        assert type(stoch_activations) is ListType, "stoch_activations must be a list: {0!r}".\
            format(stoch_activations)
        assert type(stoch_n_hidden) is ListType, "stoch_n_hidden must be a list: {0!r}".format(
            stoch_n_hidden)
        assert stoch_n_hidden == [-1] or len(stoch_n_hidden) == len(stoch_activation) - 1, \
            "len(stoch_n_hidden) must be len(stoch_activations) -1 or stoch_n_hidden = [-1]."\
            " stoch_n_hidden = {0!r} and stoch_activations = {1!r}".format(stoch_n_hidden,
                                                                           stoch_activations)
        assert type(likelihood_precision) is IntType or FloatType, "precision must be int or float: {0!r}".\
            format(likelihood_precision)

        assert type(batch_normalization) is bool, "Batch normalization must be boolean. Provided: {0!r}".format(
            batch_normalization)

        self.n_hidden = np.array(n_hidden)
        self.n_out = n_out
        self.n_in = n_in
        self.stoch_n_hidden = [np.array(i) for i in stoch_n_hidden]
        self.det_activation_names = det_activations
        self.det_activation, self.det_activation_prime = parse_activations(
            det_activations)
        self.stoch_activation_names = stoch_activations
        self.stoch_activation, self.stoch_activation_prime = parse_activations(
            stoch_activations)
        self.likelihood_precision = np.asarray(
            likelihood_precision, dtype=theano.config.floatX)
        self.batch_normalization = batch_normalization

    def define_network(self, layers_info=None):
        """
        Builds Theano graph of the network.
        """
        self.hidden_layers = [None] * self.n_hidden.size

        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = self.hidden_layer_type(self.rng, self.trng, self.x, self.n_in,
                                                               h, self.det_activation[
                                                                   i],
                                                               self.stoch_n_hidden, self.stoch_activation,
                                                               det_activation_name=self.det_activation_names[
                                                                   i],
                                                               stoch_activation_names=self.stoch_activation_names,
                                                               m=self.m,
                                                               det_W=None if layers_info is None else
                                                               np.array(
                                                                   layers_info['hidden_layers'][
                                                                       i]['LBNlayer']['detLayer']
                                                                   ['W']),
                                                               det_b=None if layers_info is None else
                                                               np.array(layers_info['hidden_layers'][i]
                                                                        ['LBNlayer']['detLayer']['b']),
                                                               stoch_mlp_info=None if layers_info is None else
                                                               layers_info['hidden_layers'][i][
                                                                   'LBNlayer']['stochLayer'],
                                                               timeseries_layer=self.timeseries_network,
                                                               batch_normalization=self.batch_normalization,
                                                               batch_normalization_info=None if layers_info is None or 'batch_normalization' not in layers_info[
                                                                   'hidden_layers'][i]['LBNlayer'].keys() else layers_info['hidden_layers'][i]['LBNlayer']['batch_normalization'])

            else:
                self.hidden_layers[i] = self.hidden_layer_type(self.rng, self.trng,
                                                               self.hidden_layers[
                                                                   i - 1].output,
                                                               self.n_hidden[
                                                                   i - 1], h, self.det_activation[i],
                                                               self.stoch_n_hidden, self.stoch_activation,
                                                               det_activation_name=self.det_activation_names[
                                                                   i],
                                                               stoch_activation_names=self.stoch_activation_names,
                                                               det_W=None if layers_info is None else
                                                               np.array(layers_info['hidden_layers'][i]['LBNlayer']
                                                                        ['detLayer']['W']),
                                                               det_b=None if layers_info is None else
                                                               np.array(layers_info['hidden_layers'][i]['LBNlayer']
                                                                        ['detLayer']['b']),
                                                               stoch_mlp_info=None if layers_info is None else
                                                               layers_info['hidden_layers'][i][
                                                                   'LBNlayer']['stochLayer'],
                                                               timeseries_layer=self.timeseries_network,
                                                               batch_normalization=self.batch_normalization,
                                                               batch_normalization_info=None if layers_info is None or 'batch_normalization' not in layers_info[
                                                                   'hidden_layers'][i]['LBNlayer'].keys() else layers_info['hidden_layers'][i]['LBNlayer']['batch_normalization'])

            self.params.append(self.hidden_layers[i].params)

        if self.with_output_layer:
            if not self.timeseries_network:
                self.output_layer = self.output_layer_type(self.rng, self.hidden_layers[-1].output,
                                                           self.n_hidden[-1],
                                                           self.n_out, self.det_activation[
                                                           -1],
                                                           self.det_activation_names[
                                                           -1],
                                                           V_values=None
                                                           if layers_info is None else np.array(
                    layers_info['output_layer']['LBNlayer']
                    ['W']),
                    timeseries_layer=self.timeseries_network,
                    batch_normalization=self.batch_normalization,
                    gamma_values=None if layers_info is None or 'batch_normalization' not in layers_info['output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer'][
                    'batch_normalization']['gamma_values'],
                    beta_values=None if layers_info is None or 'batch_normalization' not in layers_info['output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer'][
                    'batch_normalization']['beta_values'],
                    epsilon=1e-12 if layers_info is None or 'batch_normalization' not in layers_info['output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer'][
                    'batch_normalization']['epsilon'],
                    fixed_means=None if layers_info is None or 'batch_normalization' not in layers_info['output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer'][
                    'batch_normalization']['fixed_means'],
                    stdb=None if layers_info is None or 'batch_normalization' not in layers_info['output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer'][
                    'batch_normalization']['stdb'],
                    mub=None if layers_info is None or 'batch_normalization' not in layers_info['output_layer'][
                    'LBNlayer'].keys() else layers_info['output_layer']['LBNlayer']['batch_normalization']['mub'])

            else:
                self.output_layer = self.hidden_layer_type(self.rng, self.trng,
                                                           self.hidden_layers[
                                                               -1].output,
                                                           self.n_hidden[
                                                               -1], self.n_out, self.det_activation[-1],
                                                           self.stoch_n_hidden, self.stoch_activation,
                                                           det_activation_name=self.det_activation_names[
                                                               -1],
                                                           stoch_activation_names=self.stoch_activation_names,
                                                           det_W=None if layers_info is None else
                                                           np.array(layers_info['output_layer']['LBNlayer']
                                                                    ['detLayer']['W']),
                                                           det_b=None if layers_info is None else
                                                           np.array(layers_info['output_layer']['LBNlayer']
                                                                    ['detLayer']['b']),
                                                           stoch_mlp_info=None if layers_info is None else
                                                           layers_info['output_layer'][
                                                               'LBNlayer']['stochLayer'],
                                                           timeseries_layer=self.timeseries_network,
                                                           batch_normalization=self.batch_normalization,
                                                           batch_normalization_info=None if layers_info is None or 'batch_normalization' not in layers_info[
                                                               'output_layer']['LBNlayer'].keys() else layers_info['output_layer']['LBNlayer']['batch_normalization'])

            self.params.append(self.output_layer.params)
            self.output = self.output_layer.output

        else:
            self.output = self.hidden_layers[-1].output

    def fiting_variables(self, batch_size, train_set_x, train_set_y, test_set_x=None):
        """Sets useful variables for locating batches"""
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
        # And weight the reported error by the batch_size when we avrage
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

    def fit(self, x, y, m, learning_rate, epochs, batch_size, fname=None, save_every=1, epoch0=1):
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
        self.log.info("Fit starts: m: {0}, learning_rate: {1}, epochs: {2}, batch_size: {3}, "
                      "fname: {4}, save_every: {5}".format(
                          m, learning_rate, epochs, batch_size, fname, save_every))
        train_set_x = theano.shared(np.asarray(x,
                                               dtype=theano.config.floatX))

        train_set_y = theano.shared(np.asarray(y,
                                               dtype=theano.config.floatX))

        self.fiting_variables(batch_size, train_set_x, train_set_y)

        flat_params = [p for layer in self.params for p in layer]
        gparams = [T.grad(-1. / x.shape[0] * self.log_likelihood, p)
                   for p in flat_params]

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
        if fname is not None:
            path_name, file_name = os.path.split(fname)

            if not os.path.exists("{0}/networks".format(path_name)):
                os.makedirs("{0}/networks".format(path_name))

            if not os.path.exists("{0}/likelihoods".format(path_name)):
                os.makedirs("{0}/likelihoods".format(path_name))
        for e in xrange(epoch0, epochs + epoch0):
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_likelihood = self.train_model(
                    minibatch_idx, self.n_train)

            log_likelihood.append(self.get_log_likelihood(x, y, m))

            epoch_message = "Epoch {0} log likelihood: {1}".format(
                e, log_likelihood[-1])
            self.log.info(epoch_message)
            if fname is not None:
                if e % save_every == 0 or e == epochs + epoch0 - 1:

                    self.save_network(
                        "{0}/networks/{1}_epoch_{2}.json".format(path_name, file_name, e))
                    self.log.info("Network saved.")

                    with open('{0}/likelihoods/{1}.csv'.format(path_name, file_name), 'a') as f:
                        for i, l in enumerate(
                                log_likelihood[e - epoch0 - save_every + 1:e - epoch0 + 1]):
                            f.write('{0},{1}\n'.format(
                                e - save_every + i + 1, l))

    def generate_saving_string(self):

        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_hidden": self.n_hidden.tolist(),
                                     "n_out": self.n_out,
                                     "det_activations": self.det_activation_names,
                                     "stoch_activations": self.stoch_activation_names,
                                     "stoch_n_hidden": [sh.tolist() for sh in self.stoch_n_hidden],
                                     "timeseries_network": self.timeseries_network,
                                     "likelihood_precision": self.likelihood_precision.tolist(),
                                     "hidden_layer_type": self.hidden_layer_type_name,
                                     "output_layer_type": self.output_layer_type_name,
                                     "batch_normalization": self.batch_normalization,
                                     "with_output_layer": self.with_output_layer})

        output_string += ",\"layers\":{\"hidden_layers\":["
        for k, l in enumerate(self.hidden_layers):
            det = l.det_layer
            stoch = l.stoch_layer
            if k > 0:
                output_string += ","
            output_string += "{\"LBNlayer\":{\"detLayer\":"
            output_string += json.dumps({"n_in": det.n_in, "n_out": det.n_out,
                                         "activation": det.activation_name, "W": det.W.get_value().tolist(),
                                         "b": det.b.get_value().tolist() if det.no_bias is False else None,
                                         "no_bias": det.no_bias, "timeseries": det.timeseries})
            output_string += ", \"stochLayer\":"
            output_string += "["
            for i, hs in enumerate(stoch.hidden_layers):
                if i > 0:
                    output_string += ","
                output_string += "{\"detLayer\":"
                output_string += json.dumps({"n_in": hs.n_in, "n_out": hs.n_out,
                                             "activation": hs.activation_name, "W": hs.W.get_value().tolist(),
                                             "b": hs.b.get_value().tolist() if hs.no_bias is False else None,
                                             "no_bias": hs.no_bias,
                                             "timeseries": hs.timeseries})
                output_string += "}"
            output_string += "]"
            if self.batch_normalization:
                output_string += ", \"batch_normalization\":{\"detLayer\":"

                output_string += json.dumps({"gamma_values":
                                             det.gamma.get_value().tolist(),
                                             "beta_values":
                                             det.beta.get_value().tolist(),
                                             "epsilon":
                                             det.epsilon})
                output_string += ", \"stochLayer\":"
                output_string += "["
                for i, hs in enumerate(stoch.hidden_layers):
                    if i > 0:
                        output_string += ","
                    output_string += "{\"detLayer\":"
                    output_string += json.dumps({"gamma_values":
                                                 hs.gamma.get_value().tolist(),
                                                 "beta_values":
                                                 hs.beta.get_value().tolist(),
                                                 "epsilon":
                                                 hs.epsilon})
                    output_string += "}"
                output_string += "]}"

            output_string += "}}"
        output_string += "]"

        if self.with_output_layer:
            output_string += ",\"output_layer\":{\"LBNlayer\":"

            if self.timeseries_network:
                det = self.output_layer.det_layer
                stoch = self.output_layer.stoch_layer
                output_string += "{\"detLayer\":"
                output_string += json.dumps({"n_in": det.n_in, "n_out": det.n_out,
                                             "activation": det.activation_name, "W": det.W.get_value().tolist(),
                                             "b": det.b.get_value().tolist()if det.no_bias is False else None,
                                             "no_bias": det.no_bias})
                output_string += ", \"stochLayer\":"
                output_string += "["
                for i, hs in enumerate(stoch.hidden_layers):
                    if i > 0:
                        output_string += ","
                    output_string += "{\"detLayer\":"
                    output_string += json.dumps({"n_in": hs.n_in, "n_out": hs.n_out,
                                                 "activation": hs.activation_name, "W": hs.W.get_value().tolist(),
                                                 "b": hs.b.get_value().tolist() if hs.no_bias is False else None,
                                                 "no_bias": hs.no_bias,
                                                 "timeseries": hs.timeseries})
                    output_string += "}"
                output_string += "]"
                if self.batch_normalization:
                    output_string += ", \"batch_normalization:\": {\"detLayer\":"
                    output_string += json.dumps({"gamma_values":
                                                 det.gamma.get_value().tolist(),
                                                 "beta_values":
                                                 det.beta.get_value().tolist(),
                                                 "epsilon":
                                                 det.epsilon})
                    output_string += ", \"stochLayer\":"
                    output_string += "["
                    for i, hs in enumerate(stoch.hidden_layers):
                        if i > 0:
                            output_string += ","
                        output_string += "{\"detLayer\":"
                        output_string += json.dumps({"gamma_values":
                                                     hs.gamma.get_value().tolist(),
                                                     "beta_values":
                                                     hs.beta.get_value().tolist(),
                                                     "epsilon":
                                                     hs.epsilon})
                        output_string += "}"

                    output_string += "]}"

                output_string += "}"
            else:
                output_dict = {"n_in": self.output_layer.n_in,
                               "n_out": self.output_layer.n_out,
                               "activation": self.output_layer.activation_name,
                               "W": self.output_layer.W.get_value().tolist(),
                               "timeseries": self.output_layer.timeseries}
                if self.batch_normalization:
                    output_dict[
                        'gamma_values'] = self.output_layer.gamma.get_value().tolist()
                    output_dict[
                        'beta_values'] = self.output_layer.beta.get_value().tolist()
                    output_dict['epsilon'] = self.output_layer.epsilon

                output_string += json.dumps(output_dict)
            output_string += "}}}"
        else:
            output_string += "}}"
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
    def init_from_file(cls, fname, log=None, session_name=None):
        """
        Loads a saved network from file fname.
        :type fname: string.
        :param fname: file name (with local or global path) from where to load the network.
        """
        with open(fname) as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_lbn = cls(network_properties['n_in'], network_properties['n_hidden'],
                         network_properties['n_out'], network_properties[
                             'det_activations'],
                         network_properties['stoch_activations'],
                         network_properties['stoch_n_hidden'],
                         layers_info=network_description['layers'],
                         log=log, session_name=session_name,
                         likelihood_precision=1 if 'likelihood_precision' not in network_properties.keys() else
                         network_properties['likelihood_precision'],
                         with_output_layer=network_properties['with_output_layer'] if 'with_output_layer' in network_properties.keys() else True)

        loaded_lbn.log.info('LBN Network loaded from file: {0}.'.format(fname))

        return loaded_lbn


class ResidualLBN(StochasticInterface):

    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                 stoch_n_hidden=[-1],
                 layers_info=None,
                 timeseries_network=False,
                 log=None,
                 input_var=None,
                 likelihood_precision=1.,
                 batch_normalization=False, with_output_layer=True,
                 m=None):
        self.hidden_layer_type = LBNHiddenLayer
        self.output_layer_type = LBNOutputLayer
        self.hidden_layer_type_name = "LBNHiddenLayer"
        self.output_layer_type_name = "LBNOutputLayer"

        super(ResidualLBN, self).__init__(n_in, n_hidden, n_out, det_activations, stoch_activations,
                                          stoch_n_hidden=stoch_n_hidden,
                                          layers_info=layers_info,
                                          timeseries_network=timeseries_network,
                                          log=log,
                                          input_var=input_var,
                                          likelihood_precision=likelihood_precision,
                                          batch_normalization=batch_normalization,
                                          with_output_layer=with_output_layer,
                                          m=m)

    def define_network(self, layers_info=None):

        super(ResidualLBN, self).define_network(layers_info=layers_info)

        Weye = T.eye(self.x.shape[1],
                     self.hidden_layers[-1].det_layer.a.shape[1])
        self.output = self.hidden_layers[-1].det_layer.activation(
            self.hidden_layers[-1].det_layer.a + T.dot(self.x, Weye)) * self.hidden_layers[-1].stoch_layer.output


class LBN(StochasticInterface):

    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                 stoch_n_hidden=[-1],
                 layers_info=None,
                 timeseries_network=False,
                 log=None,
                 input_var=None,
                 likelihood_precision=1.,
                 batch_normalization=False, with_output_layer=True):
        self.hidden_layer_type = LBNHiddenLayer
        self.output_layer_type = LBNOutputLayer
        self.hidden_layer_type_name = "LBNHiddenLayer"
        self.output_layer_type_name = "LBNOutputLayer"

        super(LBN, self).__init__(n_in, n_hidden, n_out, det_activations, stoch_activations,
                                  stoch_n_hidden=stoch_n_hidden,
                                  layers_info=layers_info,
                                  timeseries_network=timeseries_network,
                                  log=log,
                                  input_var=input_var,
                                  likelihood_precision=likelihood_precision,
                                  batch_normalization=batch_normalization, with_output_layer=with_output_layer)


class NoisyMLP(StochasticInterface):

    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                 stoch_n_hidden=[-1],
                 layers_info=None,
                 timeseries_network=False,
                 log=None,
                 input_var=None,
                 likelihood_precision=1.,
                 batch_normalization=False):
        self.hidden_layer_type = NoisyMLPHiddenLayer
        self.output_layer_type = LBNOutputLayer
        self.hidden_layer_type_name = "NoisyMLPHiddenLayer"
        self.output_layer_type_name = "LBNOutputLayer"

        super(NoisyMLP, self).__init__(n_in, n_hidden, n_out, det_activations, stoch_activations,
                                       stoch_n_hidden=stoch_n_hidden,
                                       layers_info=layers_info,
                                       timeseries_network=timeseries_network,
                                       log=log,
                                       input_var=input_var,
                                       likelihood_precision=likelihood_precision,
                                       batch_normalization=batch_normalization)
