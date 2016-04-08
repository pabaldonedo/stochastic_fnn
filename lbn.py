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
        [self.a, self.output], _ = theano.scan(h_step, sequences=[self.input])
        
        #### ONE SAMPLE
        #self.a = T.dot(self.input, self.W.T)
        #self.output = self.activation(self.a)

class DetHiddenLayer(object):
    def __init__(self, rng, input_var, n_in, n_out, activation, activation_prime,
                                                            m=None, W=None, b=None, no_bias=False):
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

    def __init__(self, rng, trng, input_var, n_in, n_hidden, n_out, activations, activation_prime):
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

        self.params = [None]*(self.n_hidden.size+1)*2

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

        self.hidden_layers[-1] = DetHiddenLayer(rng, self.hidden_layers[-2].output,
                                                    self.n_hidden[-1], self.n_out,
                                                    activations[-1], activation_prime[-1])
        self.params[-2] = self.hidden_layers[-1].W
        self.params[-1] = self.hidden_layers[-1].b

        self.ph = self.hidden_layers[-1].output
        sample = trng.uniform(size=self.ph.shape)
        epsilon = theano.gradient.disconnected_grad(T.lt(sample, self.ph) - self.ph)
        self.output = self.ph + epsilon#T.lt(sample, self.ph)
        #theano.gradient.disconnected_grad(epsilon)


class LBNHiddenLayer():

    def __init__(self, rng, trng, input_var, n_in, n_out, det_activation, det_activation_prime,
                                det_activation_name,
                                stoch_n_hidden, stoch_activations, stoch_activation_prime,
                                stoch_activation_names, m=None):
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
                                        h, self.det_activation[i], self.det_activation_prime[i],
                                        self.det_activation_names[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_prime,
                                        self.stoch_activation_names, m=self.m)
            else:
                self.hidden_layers[i] = LBNHiddenLayer(self.rng, self.trng,
                                        self.hidden_layers[i-1].output,
                                        self.n_hidden[i-1], h, self.det_activation[i],
                                        self.det_activation_prime[i],
                                        self.det_activation_names[i],
                                        self.stoch_n_hidden, self.stoch_activation,
                                        self.stoch_activation_prime,
                                        self.stoch_activation_names)

            self.params += self.hidden_layers[i].params

        self.output_layer = OutputLayer(self.rng, self.hidden_layers[-1].output,
                                                    n_hidden[-1], n_out, self.det_activation[-1])

        self.params += self.output_layer.params
        self.output = self.output_layer.output
        self.log_likelihood = T.sum(T.log(T.sum(T.exp(-0.5*T.sum((self.output - self.y.dimshuffle(
                                                                'x',0,1))**2, axis=2)), axis=0)))-\
                                    self.y.shape[0]*T.log(self.m*T.sqrt((2*np.pi)**self.y.shape[1]))

        self.tmp = T.sum(T.sum((self.output - self.y.dimshuffle('x',0,1))**2, axis=2), axis=1)

        self.tmp0 = self.tmp[0]
        self.tmp1 = self.tmp[1]
        self.tmp_all = T.sum(self.tmp)

        self.predict = theano.function(inputs=[self.x, self.m], outputs=self.output)

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

    def fit(self, x, y, m, learning_rate, epochs):
        
        def gradient_step(it, gf, output_layer_input):
        
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

            #error = (self.output[it]-self.y[it])
            #gd = T.exp(-T.sum(error**2, axis=1, keepdims=True))
            #gf = gd*2*error
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
        tmp = gparams
        gparams = [ 1./(self.x.shape[0]) *T.sum(gp, axis=0) for gp in gparams]

        params = get_params()
        #upd = [(param, param - learning_rate * gparam)
        #        for param, gparam in zip(params, gparams)]

        #self.train_model = theano.function(inputs=[self.x, self.y,self.m],
        #                                outputs=self.log_likelihood,
        #                                updates=upd)
        
        gparams_theano = [T.grad(-1./x.shape[0]*self.log_likelihood, p) for p in params]
        gparams_theano0 = [T.grad(-1./x.shape[0]*self.tmp0, p) for p in params]
        gparams_theano1 = [T.grad(-1./x.shape[0]*self.tmp1, p) for p in params]
        gparams_theano_all = [T.grad(-1./x.shape[0]*self.tmp_all, p) for p in params]

        upd = [(param, param - learning_rate * gparam)
                for param, gparam in zip(params, gparams_theano)]

        self.train_model = theano.function(inputs=[self.x, self.y,self.m],
                                        outputs=[gparams_theano0[0], gparams_theano1[0],
                                                gparams_theano_all[0], gparams_theano[0],
                                                tmp[0], self.hidden_layers[-1].output, gparams[0]],#[gparams_theano0[0], gparams_theano1[0], tmp[0]*1./self.x.shape[0], gparams_theano[0], gparams[0]],#[T.sum((gparams[j]-gparams_theano[j])**2)*1./T.sum(gparams[j]**2) for j in xrange(11)] + [gparams[0]*1./gparams_theano[0]],#self.log_likelihood,
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

        #log_likelihood = np.ones(epochs)
        for e in xrange(epochs):
         #   log_likelihood[e] = self.train_model(x,y,m)
            #print log_likelihood[e]
            ge = self.train_model(x,y,m)
            print "theano partials"
            print ge[0]
            print ge[1]
            print "theano total"
            print ge[2]
            print ge[3]
            print "partials"

            print ge[4]
            print "hidden"
            print ge[5]
            print "own total"
            print ge[6]

            import ipdb
            ipdb.set_trace()
        #import matplotlib.pyplot as plt
        #plt.plot(np.arange(epochs),log_likelihood)
        #plt.show()

        #for e in xrange(epochs):
        #    ge = self.train_model(x,y,m)
        #    print ge
        #    print "------"
            #print ge[2]
            #print ge
            #print "Gradient"
            #print ge[0]
            #print "OWN"
#            print ge[1]
            #print 2*np.dot(ge[8].T, ge[1]*ge[2]) #/ np.sqrt(np.sum(2*np.dot(ge[3].T, ge[1]*ge[2])**2))
            #print "OWN THEANO"
            #print ge[7]
            #print "Gradient:"
            #print repr(ge[0])
            #print "H"
            #print repr(ge[1])
            #print "A"
            #print repr(ge[2])
            #print "ERROR"
            #print repr(ge[3])
            #print "LIKE"
            #print repr(ge[4])
            #print "GF"
            #print repr(ge[8])
            #print "GF THEANO"
            #print repr(ge[9])
            #print "w"
            #print repr(ge[5])
            #print "x"
            #print repr(ge[6])
            #print "Wx"
            #print repr(np.dot(ge[6], ge[5].T))
            #print "------"


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
    det_activations = ['linear', 'linear']
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
    import ipdb
    ipdb.set_trace()
#    ge = n.debug_feedforward(x_train,m)
#    for g in ge:
#        print repr(g)
#    print "VARS"
#    for p in n.params[-1::-1]:
#        print repr(p.get_value())

   # x = np.random.randn(3,n_in)
    #y = np.random.randn(3,n_out)
