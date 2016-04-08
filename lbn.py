import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import types
from types import IntType
from types import ListType
from util import parse_activations



class OutputLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, V=None):
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

        :type V: numpy.array.
        :param V: initialization values of the weights.
        """
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
        [self.a, self.output], _ = theano.scan(h_step, sequences=[self.input])
        
        #### ONE SAMPLE
        #self.a = T.dot(self.input, self.W.T)
        #self.output = self.activation(self.a)

class DetHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_prime,
                                                            m=None, W=None, b=None, no_bias=False):
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

        :type m: int.
        :param m: number of samples to be drawn in the layer and the input is consider constant.
                If b is None input is treat as a scan input sequence.

        :type W: numpy.array.
        :param W: initialization values of the weights.

        :type b: numpy array.
        :param b: initialization values of the bias.

        :type no_bias: bool.
        :param no_bias: sets if the layer has bias variable or not.
        """
        self.input = input_var
        self.no_bias = no_bias

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

        if b is None and no_bias is False:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        if no_bias is False:
            self.b = b
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        self.activation = activation
        self.activation_prime = activation_prime

        ########### ONE SAMPLE
        #self.no_bias = T.dot(self.input, self.W.T)
        #if no_bias:
        #    self.a = self.no_bias
        #else:
        #    self.a = self.no_bias + self.b
        #self.output = self.activation(self.a)
        #self.delta = self.activation_prime(self.a)
        ###########
        def h_step(x):
            no_bias_output = T.dot(x, self.W.T)
            if self.no_bias:
                a = no_bias_output
            else:
                a = no_bias_output + self.b
            output = self.activation(a)
            delta = self.activation_prime(a)
            return no_bias_output, a, output, delta

        if m is None:
            [self.no_bias_output, self.a, self.output, self.delta], _ = theano.scan(
                                                                h_step, sequences=self.input)
        else:
            [self.no_bias_output, self.a, self.output, self.delta], _ = theano.scan(
                                                                h_step, non_sequences=self.input,
                                                                outputs_info=[None]*4, n_steps=m)
class StochHiddenLayer(object):
    """
    Stochastic hidden MLP that are included in each LBN hidden layer.
    """
    def __init__(self, rng, trng, input_var, n_in, n_hidden, n_out, activations, activation_prime):
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
        """

        self.input = input_var
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.hidden_layers = [None]*(self.n_hidden.size+1)

        self.params = [None]*(self.n_hidden.size+1)*2

        #Builds hidden layers of the MLP.
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.input, self.n_in, h,
                                                                activations[i], activation_prime[i])
            else:
                self.hidden_layers[i] = DetHiddenLayer(rng, self.hidden_layers[i-1].output,
                                                            self.n_hidden[i-1], h,
                                                            activations[i], activation_prime[i])
            self.params[2*i] = self.hidden_layers[i].W
            self.params[2*i+1] = self.hidden_layers[i].b

        #Output layer of MLP.
        self.hidden_layers[-1] = DetHiddenLayer(rng, self.hidden_layers[-2].output,
                                                    self.n_hidden[-1], self.n_out,
                                                    activations[-1], activation_prime[-1])
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
    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation, det_activation_prime,
                                stoch_n_hidden, stoch_activations, stoch_activation_prime,
                                det_activation_name=None, stoch_activation_names=None, m=None):
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
        """

        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.det_activation = det_activation_name
        self.det_activation_prime = det_activation_prime
        self.stoch_activation = stoch_activation_names
        self.det_layer = DetHiddenLayer(rng, input_var, n_in, n_out, det_activation,
                                                            det_activation_prime, m=m, no_bias=True)
           
        #If -1, same hidden units
        stoch_n_hidden = np.array([i if i > -1 else n_out for i in stoch_n_hidden])
        self.stoch_layer = StochHiddenLayer(rng, trng, self.det_layer.no_bias_output,
                                                    n_out, stoch_n_hidden, n_out,
                                                    stoch_activations, stoch_activation_prime)

        self.output = self.stoch_layer.output*self.det_layer.output
        self.params = self.det_layer.params + self.stoch_layer.params


class LBN:
    """
    Linearizing Belief Net (LBN) as explained in "Predicting Distributions with
    Linearizing Belief Networks" paper by Yann N. Dauphin and David Grangie from Facebook AI
    Research.
    """
    def __init__(self, n_in, n_hidden, n_out, det_activations, stoch_activations,
                                                                            stoch_n_hidden=[-1]):
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
        """
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.trng = T.shared_randomstreams.RandomStreams(1234)
        self.rng = np.random.RandomState(0)
        self.m = T.iscalar('M') 
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

    def define_network(self):
        """
        Builds Theano graph of the network.
        """
        self.hidden_layers = [None]*self.n_hidden.size
        self.params = []
        for i, h in enumerate(self.n_hidden):
            if i == 0:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng, self.x, self.n_in,
                                        h, self.det_activation[i], self.det_activation_prime[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_prime,
                                        det_activation_name=self.det_activation_names[i],
                                        stoch_activation_names=self.stoch_activation_names,
                                        m=self.m)
            else:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng,
                                        self.hidden_layers[i-1].output,
                                        self.n_hidden[i-1], h, self.det_activation[i],
                                        self.det_activation_prime[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_prime,
                                        det_activation_name=self.det_activation_names[i],
                                        stoch_activation_names=self.stoch_activation_names)

            self.params += self.hidden_layers[i].params

        self.output_layer = OutputLayer(self.rng, self.hidden_layers[-1].output,
                                                    n_hidden[-1], n_out, self.det_activation[-1])

        self.params += self.output_layer.params
        self.output = self.output_layer.output
        self.log_likelihood = T.sum(T.log(T.sum(T.exp(-0.5*T.sum((self.output - self.y.dimshuffle(
                                                                'x',0,1))**2, axis=2)), axis=0)))-\
                                    self.y.shape[0]*T.log(self.m*T.sqrt((2*np.pi)**self.y.shape[1]))

        self.predict = theano.function(inputs=[self.x, self.m], outputs=self.output)


        ###Debugging of the feedforward pass.#####
        debug_list = []
        debug_list.append(self.output)
        for h in self.hidden_layers:
            debug_list.append(h.output)
            debug_list.append(h.stoch_layer.output)

            sublist = [hi.output for hi in h.stoch_layer.hidden_layers[-1::-1]]
            debug_list += sublist
            debug_list.append(h.det_layer.output)

        self.debug_feedforward = theano.function(inputs=[self.x, self.m],
                                                outputs=debug_list + [self.x])


    def fit(self, x, y, m, learning_rate, epochs):
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
        """
        
        def gradient_step(it, gf, output_layer_input):
            """
            Computes the gradient for a sample. It only works for one sample as theano resamples
            every time the graph values after each it.
            """
            def stochastic_gradient(layer, gha, it):
                gparams = []
                for i, h in enumerate(layer.hidden_layers[-1::-1]):
                    ga = gha * h.delta[it]
                    #intermediate_result = gha*h.delta[it]
                    gw = T.dot(ga.T, h.input[it]) #T.dot(gb, h.input.T)
                    gb = T.sum(ga, axis=0)#gha*h.delta
                    gparams.append(gw)
                    gparams.append(gb)
                    gha = T.dot(ga, h.W)#h.delta*T.dot(h.W.T, gha)

                return gparams, gha

            gv = T.dot(gf.T, output_layer_input)# self.output_layer.input[it])#T.dot(gf, self.output_layer.input.T)
            gparams = []
            gparams.append(gv)
            for i, h_layer in enumerate(self.hidden_layers[-1::-1]):

                a = h_layer.det_layer.output[it]
                h = h_layer.stoch_layer.output[it]
                if i == 0:
                    gh = a*T.dot(gf, self.output_layer.W)#a*T.dot(self.output_layer.W.T, gf)
                else:
                    previous_layer = self.hidden_layers[len(self.n_hidden)-i]
                    gh = a*T.dot(ga, previous_layer.det_layer.W)#a*T.dot(previous_layer.det_layer.W.T, ga)

                gp, gha = stochastic_gradient(h_layer.stoch_layer, gh, it)

                if i==0:
                    ga = h*T.dot(gf, self.output_layer.W) + gha#h*T.dot(self.output_layer.W.T, gf) + gha
                else:
                    ga = h*T.dot(ga, previous_layer.det_layer.W) + gha#h*T.dot(previous_layer.det_layer.W.T, ga) + gha

                gparams += gp
                if i == len(self.hidden_layers)-1:
                    gw = T.dot(ga.T, h_layer.input)#T.dot(ga, h_layer.input.T)
                else:
                    gw = T.dot(ga.T, h_layer.input[it])
                gparams.append(gw)

            return gparams

        def get_params():
            def get_stochastic_params(layer):
                params = []
                for i, h in enumerate(layer.hidden_layers[-1::-1]):
                    params.append(h.W)
                    params.append(h.b)
                return params

            params = []
            params.append(self.output_layer.W)
            for i, h_layer in enumerate(self.hidden_layers[-1::-1]):
                params += get_stochastic_params(h_layer.stoch_layer)
                params.append(h_layer.det_layer.W)

            return params
        
        error = self.output-self.y.dimshuffle('x',0,1)
        aux = T.exp(-T.sum(error**2, axis=2, keepdims=True))
        gd = aux*1./T.sum(aux, axis=0, keepdims=True)
        gf = error * gd

        gparams, _ = theano.scan(gradient_step, sequences=[T.arange(self.m), gf, self.output_layer.input])
        gparams = [ 1./(self.x.shape[0]) *T.sum(gp, axis=0) for gp in gparams]

        params = get_params()
        #upd = [(param, param - learning_rate * gparam)
        #        for param, gparam in zip(params, gparams)]

        #self.train_model = theano.function(inputs=[self.x, self.y,self.m],
        #                                outputs=self.log_likelihood,
        #                                updates=upd)
        
        gparams_theano = [T.grad(-1./x.shape[0]*self.log_likelihood, p) for p in params]

        upd = [(param, param - learning_rate * gparam)
                for param, gparam in zip(params, gparams_theano)]

        self.train_model = theano.function(inputs=[self.x, self.y,self.m],
                                        outputs=self.log_likelihood,#[gparams_theano0[0], gparams_theano1[0], tmp[0]*1./self.x.shape[0], gparams_theano[0], gparams[0]],#[T.sum((gparams[j]-gparams_theano[j])**2)*1./T.sum(gparams[j]**2) for j in xrange(11)] + [gparams[0]*1./gparams_theano[0]],#self.log_likelihood,
                                        updates=upd)

                                                #T.grad(-self.log_likelihood,
                                                #        self.hidden_layers[-1].stoch_layer.hidden_layers[-1].W)],
                                                #[gparams[0], gparams_theano[0]],
                                                #self.hidden_layers[-1].stoch_layer.output,
                                                #self.hidden_layers[-1].det_layer.output,
                                                #error, self.tmp_likelihood,
                                                #self.hidden_layers[0].det_layer.W,
                                                #self.x, gparams[0], gf,
                                                #T.grad(-self.tmp_likelihood, self.output_layer.output)],

        log_likelihood = np.ones(epochs)
        for e in xrange(epochs):
            log_likelihood[e] = self.train_model(x,y,m)
            print log_likelihood[e]
        plt.plot(np.arange(epochs),log_likelihood)
        plt.show()


if __name__ == '__main__':

    import cPickle, gzip, numpy
    np.random.seed(0)
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    x_train = train_set[0]
    y_train = train_set[1]
    y_val = valid_set[1]
    f.close()


    x_train = x_train[:1,:].reshape(1,-1)
    y_train = y_train[:1].reshape(1,-1)

    n_in = x_train.shape[1]
    n_hidden = [5, 4]
    n_out = 1
    det_activations = ['linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    stoch_n_hidden = [-1]
    m = 2
    n = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)
    n.fit(x_train,y_train,m,.01,50)
    y_hat = n.predict(x_train[0].reshape(1,-1), m)
    y_points = np.linspace(-1,10).reshape(1,-1,1)

    distribution = np.sum(np.exp(-0.5*np.sum((y_points-y_hat)**2, axis=2)), axis=0)*1./(m*np.sqrt((2*np.pi)**y_hat.shape[2]))
    import matplotlib.pyplot as plt
    plt.plot(y_points[0,:,0], distribution)
    plt.show()

#    ge = n.debug_feedforward(x_train,m)
#    for g in ge:
#        print repr(g)
#    print "VARS"
#    for p in n.params[-1::-1]:
#        print repr(p.get_value())

   # x = np.random.randn(3,n_in)
    #y = np.random.randn(3,n_out)
