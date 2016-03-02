import theano
import theano.tensor as T
import numpy as np
from util import parse_activations


class SFN:
    def __init__(self, n_in, n_hidden, n_out, activations):
        self.x = T.matrix('x')
        self.y = T.vector('y')
        self.trng = T.shared_randomstreams.RandomStreams(1234)
        self.parse_properties(n_in, n_hidden, n_out, activations)
        self.init_weights()
        self.define_network()


    def parse_properties(self, n_in, n_hidden, n_out, activations):
        self.n_hidden = np.array(n_hidden)
        self.n_out = n_out
        self.n_in = n_in
        self.activation, self.activation_prime = parse_activations(activations)

    def init_weights(self):

        theta_shape = self.n_hidden[0]*self.n_in + np.sum(self.n_hidden[1:]**2) +\
                     self.n_hidden[-1]*self.n_out + np.sum(self.n_hidden) + self.n_out
        #self.theta = theano.shared(np.asarray(np.random.uniform(size=theta_shape, low=-0.01,
        #                                high=0.01), dtype=theano.config.floatX), name='theta')
        self.theta = theano.shared(0.05*np.ones(theta_shape, dtype=theano.config.floatX), name='theta')

        self.wf = [None]*self.n_hidden.size
        param_idx = 0
        for i in xrange(self.n_hidden.size):
            if i == 0:
                h = self.n_hidden[i]*self.n_in
                self.wf[i] = self.theta[param_idx:(param_idx + h)].reshape(
                                                                    (self.n_hidden[0], self.n_in))

            else:
                h = self.n_hidden[i-1]*self.n_hidden[i]
                self.wf[i] = self.theta[param_idx:(param_idx + h)].reshape(
                                                            (self.n_hidden[i-1]), self.n_hidden[i])
            param_idx += h

        h = self.n_hidden[-1]*self.n_out
        self.w_out = self.theta[param_idx:(param_idx + h)].reshape((self.n_out, self.n_hidden[-1]))
        param_idx += h

        self.bf = [None]*self.n_hidden.size
        for i, h in enumerate(self.n_hidden):
            self.bf[i] = self.theta[param_idx:(param_idx + h)]
            param_idx += h

        h = self.n_out
        self.b_out = self.theta[param_idx:(param_idx + h)]
        param_idx += h
        assert param_idx == self.theta.shape.eval()

    def define_network(self):

        def step(ph):
            sample = self.trng.uniform(size=ph.shape)
            return  T.lt(sample, ph)

        self.h = [None]*self.n_hidden.size
        self.a = [None]*self.n_hidden.size
        self.ph = [None]*self.n_hidden.size
        self.m = T.iscalar('M')
        for i in xrange(self.n_hidden.size):
            if i == 0:
                self.a[i] = T.dot(self.x, self.wf[i].T) + self.bf[i]
            else:
                self.a[i] = T.dot(h[i-1], self.wf[i].T) + self.bf[i]
            
            self.ph[i] = self.activation[i](self.a[i])
            self.h[i], _ = theano.scan(step, non_sequences=self.ph[i].flatten(), outputs_info=None, n_steps=self.m)
       
        self.o = T.dot(self.h[-1], self.w_out.T) + self.b_out
        self.pyi = self.activation[-1](self.o)
        self.py = T.prod(self.y*self.pyi + (1-self.y)*self.pyi, axis=1)

        sample = self.trng.uniform(size=self.pyi.shape)
        self.y_pred = T.lt(sample, self.pyi)

        self.predict = theano.function(inputs=[self.x, self.m], outputs=self.y_pred, on_unused_input='warn')
        self.rnn_get_state = theano.function(inputs=[self.x, self.m], outputs=[sample, self.a[0], self.ph[0], self.h[0], self.o, self.pyi, self.y_pred])

    def fit(self, x, y, m, gradient_type, learning_rate, epochs):

        l_r = T.scalar('lr')
        self.c = -T.log(1./self.m*T.sum(self.py, axis=0))
        self.get_c = theano.function(inputs=[self.x, self.y, self.m], outputs=self.c)
        if gradient_type == 'g2':
            go, g = self.get_gradient_2()
        elif gradient_type == 'g3':
            go, g = self.get_gradient_3()
        elif gradient_type == 'g4':
            g = self.get_gradient_4()
        elif gradient_type == 'g5':
            g = self.get_gradient_5()

        updates = []
        if gradient_type in ['g2', 'g3']:
            updates.append((self.w_out, self.w_out - l_r*T.dot(go, self.h[0].T)))
            updates.append((self.b_out, self.b_out - l_r*go))
            updates.append((self.wf, self.wf - l_r*T.dot(g, self.x.T)))
            updates.append((self.bf, self.bf - l_r*g, self.h[0].T))
        else:
            updates.append((self.theta, self.theta - l_r*g))

        train_model = theano.function(inputs=[self.x, self.y, self.m],
            outputs=self.c,
            updates=updates,
            givens={l_r: learning_rate})

        for e in xrange(epochs):
            c = 0
            for i in xrange(x.shape[0]):
                c += train_model(x[i].reshape(1,-1), y[i], m)
                if (i + 1) %100 == 0:
                    print i+1
                    self.print_weights()
                if i+1 == 1000:
                    break
            print "Epoch {0}: error: {1}".format(e, c)

    def gradient_pyh(self):
        return T.grad(self.c, self.o)

    def get_gradient_2(self):
        go = self.gradient_pyh()
        g2 = T.dot(go, self.w_out)
        return go, g2


    def get_gradient_3(self):
        """Only for 1 hidden layer"""
        go, g2 = self.get_gradient_2()
        return go, self.activation_prime[0](self.a[0])*g2


    def get_gradient_4(self):
        w_mean = self.py*1./T.sum(self.py)
        phm = (T.dot(self.h[0], self.ph[0].T) + T.dot(1-self.h[0], 1-self.ph[0].T)).flatten()
        f = T.dot(w_mean, T.log(self.py) + T.log(phm))
        return T.grad(f, self.theta)

    def get_gradient_5(self):
        w_mean = self.py*1./T.sum(self.py)
        phm = (T.dot(self.h[0], self.ph[0].T) + T.dot(1-self.h[0], 1-self.ph[0].T)).flatten()
        logphm = T.log(phm)
        f = T.dot(w_mean, T.log(self.py) + logphm) - 1./m*T.sum(logphm)
        self.test = theano.function(inputs=[self.x, self.m], outputs=T.grad(f, self.theta))

        return T.grad(f, self.theta)


    def print_weights(self):
        print "----W_f----"
        for i, w in enumerate(self.wf):
            print "hidden layer number {0}".format(i)
            print w.eval()
        print "----b_f----"
        for i, b in enumerate(self.bf):
            print "hidden layer number {0}".format(i)
            print b.eval()
        print "---W_out----"
        print self.w_out.eval()

        print "---b_out----"
        print self.b_out.eval()

sfn = SFN(4,[3],2,['sigmoid', 'sigmoid'])
x = np.array([[1.5, 0, -1, 5]])
m = 4
gradient_type = 'g4'
learning_rate = 0.1
epochs = 1
# print "---x----"
# print x
# print x.shape
# print "weights"
# sfn.print_weights()
# print "-----network values----"
s, a, ph, h, o, py, y_pred = sfn.rnn_get_state(x, m)

print "s"
print s
print "a"
print a
print "p(h|x)"
print ph
print "samples h"
print h
print "o"
print o
print "p(y|h)"
print py

# print "y_pred"
# print y_pred

#sfn.fit(x,m, gradient_type, learning_rate, epochs)
#sfn.get_gradient_5()
#print sfn.test(x, m)
print "-----cost------"
#print sfn.get_c(x, m)
# sfn.gradient_3()
# go, g3 = sfn.test(x, m)
# print "---gradient g3---"
# print g3

import cPickle, gzip, numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
x_train = train_set[0]
y_train = train_set[1]
x_val = valid_set[0]
y_val = valid_set[1]
f.close()

y_train = y_train == 4
y_val = y_val == 4
sfn = SFN(x_train.shape[1],[500], 1,['sigmoid', 'sigmoid'])

sfn.fit(x_train, y_train.reshape(-1,1), m, gradient_type, learning_rate, epochs)
y_pred_train = -1*np.ones((y_train.shape[0], m))
import ipdb
ipdb.set_trace()
for i in xrange(y_pred_train.shape[0]):
    y_pred_train[i] = sfn.predict(x_train[i].reshape(1,-1), m).flatten()

y_pred_val = -1*np.ones(y_val.shape[0], m)
for i in xrange(y_pred_val[0]):
    y_pred_val[i] = sfn.predict_((x_val[i].reshape(1,-1), m)).flatten()

print np.sum(y_train*y_pred_train)

print np.sum(y_val*y_pred_val)
import ipdb
ipdb.set_trace()




