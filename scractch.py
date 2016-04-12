import theano
import theano.tensor as T
import numpy as np
from util import parse_activations


class SFN:
    def __init__(self, n_in, n_hidden, n_out, activations):
        self.x = T.matrix('x')
        self.y = T.matrix('y')
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

        theta_shape = self.n_hidden[0]*self.n_in + np.sum(self.n_hidden[:-1]*self.n_hidden[1:]) +\
                     self.n_hidden[-1]*self.n_out + np.sum(self.n_hidden) + self.n_out
        self.theta = theano.shared(np.asarray(np.random.uniform(size=theta_shape, low=-0.01,
                                        high=0.01), dtype=theano.config.floatX), name='theta')
        #self.theta = theano.shared(0.05*np.ones(theta_shape, dtype=theano.config.floatX), name='theta')
        self.theta_idx = {}

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
                                                            (self.n_hidden[i], self.n_hidden[i-1]))
            self.theta_idx['wf[{0}]'.format(i)] = (param_idx,(param_idx + h))
            param_idx += h

        h = self.n_hidden[-1]*self.n_out
        self.w_out = self.theta[param_idx:(param_idx + h)].reshape((self.n_out, self.n_hidden[-1]))
        self.theta_idx['w_out'] = (param_idx, param_idx+h)
        param_idx += h

        self.bf = [None]*self.n_hidden.size
        for i, h in enumerate(self.n_hidden):
            self.bf[i] = self.theta[param_idx:(param_idx + h)]
            self.theta_idx['bf[{0}]'.format(i)] = (param_idx, param_idx+h)
            param_idx += h

        h = self.n_out
        self.b_out = self.theta[param_idx:(param_idx + h)]
        self.theta_idx['b_out'] = (param_idx, param_idx+h)

        param_idx += h
        assert param_idx == self.theta.shape.eval()

    def define_network(self):

        def h_step():
            h = [None]*self.n_hidden.size
            a = [None]*self.n_hidden.size
            ph_prime = [None]*self.n_hidden.size
            ph = [None]*self.n_hidden.size
            for i in xrange(self.n_hidden.size):
                if i == 0:
                    a[i] = T.dot(self.x, self.wf[i].T) + self.bf[i]
                else:
                    a[i] = T.dot(h[i-1], self.wf[i].T) + self.bf[i]
                
                ph[i] = self.activation[i](a[i])
                ph_prime[i] = self.activation_prime[i](a[i])
                sample = self.trng.uniform(size=ph[i].shape)
                h[i] = T.lt(sample, ph[i])

            o = T.dot(h[-1], self.w_out.T) + self.b_out
            pyi = self.activation[-1](o)
            pyi_prime = self.activation_prime[-1](o)
            sample = self.trng.uniform(size=pyi.shape)
            y_pred = T.lt(sample, pyi)
            return a + ph + h  + [o, pyi, pyi_prime, y_pred] + ph_prime

        def delta(pyi,y):
            return y*pyi + (1-y)*(1-pyi)

        def py(pyi, y):
            return T.prod(y*pyi + (1-y)*(1-pyi), axis=1)

        self.m = T.iscalar('M') 

        network_state, _ = theano.scan(
                    h_step, outputs_info=[None]*(self.n_hidden.size*4+4), n_steps=self.m)


        self.a = network_state[:self.n_hidden.size]
        self.ph = network_state[self.n_hidden.size:(2*self.n_hidden.size)]
        self.h = network_state[(2*self.n_hidden.size):(3*self.n_hidden.size)]
        self.o = network_state[3*self.n_hidden.size]
        self.pyi = network_state[3*self.n_hidden.size + 1]
        self.pyi_prime = network_state[3*self.n_hidden.size + 2]
        self.y_pred = network_state[3*self.n_hidden.size + 3]
        self.ph_prime = network_state[3*self.n_hidden.size + 4: 4*self.n_hidden.size + 4]

        self.py, _ = theano.scan(py, sequences=self.pyi, non_sequences=self.y)
        self.delta_o, _ = theano.scan(delta, sequences=self.pyi, non_sequences=self.y)
        self.delta_o_prime, _ = theano.scan(delta, sequences=self.pyi_prime, non_sequences=self.y)

        self.delta_a = []
        self.delta_a_prime = []
        for i in xrange(self.n_hidden.size):
            delta_a, _ = theano.scan(delta, sequences=[self.ph[i], self.h[i]])
            self.delta_a.append(delta_a)
            delta_a_prime, _ = theano.scan(delta, sequences=[self.ph_prime[i], self.h[i]])
            self.delta_a_prime.append(delta_a_prime)


        self.cm = -T.log(T.mean(self.py, axis=0))
        self.c = T.sum(self.cm)
        self.get_c = theano.function(inputs=[self.x, self.y, self.m], outputs=self.c)
        #T.grad(T.mean(self.pyi[0]),self.o[0])

        #print theano.printing.debugprint(self.cm)
        tmp_out = [i for i in self.a] + [i for i in self.ph] + [i for i in self.h] + [self.o, self.pyi, self.pyi_prime, self.y_pred, self.py, self.delta_o, self.delta_o_prime, self.cm, self.c]
        self.tmp = theano.function(inputs=[self.x, self.y, self.m], outputs=tmp_out, on_unused_input='warn')
        #self.tmp2 = theano.function(inputs=[self.x, self.y, self.m], outputs=[self.wf[0], T.dot(self.x, self.wf[0].T), self.x], on_unused_input='ignore')
        
    def fit(self, x, y, m, gradient_type, learning_rate, epochs):

        l_r = T.scalar('lr')

        if gradient_type == 'g2':
            go, g = self.get_gradient_2()
        elif gradient_type == 'g3':
            go, g = self.get_gradient_3()
        elif gradient_type == 'g4':
            go, g, w_norm = self.get_gradient_4()
            f = theano.function(inputs=[self.x, self.y, self.m], outputs=[self.delta_a_prime[0].shape, self.delta_a[0].shape, w_norm.shape], on_unused_input='warn')
            print f(x,y,m)
            import ipdb
            ipdb.set_trace()
        elif gradient_type == 'g5':
            go, g = self.get_gradient_5()

        updates = []
        def w_upd(g, h):
            return 1./g.shape[0]*T.dot(g.T, h)

        w_out_upd, _ = theano.scan(w_upd, sequences=[go, self.h[-1]])
        b_out_upd = T.mean(go, axis=1)
        theta_upd = T.alloc(np.zeros(self.theta.get_value().shape), self.theta.get_value().size)

        th_idx = self.theta_idx['w_out']
        theta_upd = T.set_subtensor(theta_upd[th_idx[0]:th_idx[1]], (self.w_out - l_r*T.sum(w_out_upd, axis=0)).flatten())
        th_idx = self.theta_idx['b_out']
        theta_upd = T.set_subtensor(theta_upd[th_idx[0]:th_idx[1]], self.b_out - l_r*T.sum(b_out_upd, axis=0))

        #updates.append((theta, self.theta[self.theta_idx['w_out'][0]:self.theta_idx['w_out'][1]], (self.w_out - l_r*T.sum(w_out_upd, axis=0)).flatten()))
        #updates.append((self.b_out, self.b_out - l_r*T.sum(b_out_upd, axis=0)))
        param_idx = 0
        for i, hi in enumerate(self.n_hidden):
            th_idx = self.theta_idx['wf[{0}]'.format(i)]

            if i==0:
                wf_upd, _ = theano.scan(w_upd, sequences=g[:,:,:self.n_hidden[0]], non_sequences=self.x)
                #updates.append((self.wf[i], self.wf[i] -l_r*T.sum(wf_upd, axis=0)))

                theta_upd = T.set_subtensor(theta_upd[th_idx[0]:th_idx[1]], (self.wf[i] -l_r*T.sum(wf_upd, axis=0)).flatten())
            else:
                #tmp = theano.function(inputs=[self.x, self.y, self.m], outputs=[g.shape, g[:,:,param_idx:param_idx + hi].shape, self.h[i-1].shape])
                wf_upd, _ = theano.scan(w_upd, sequences=[g[:,:,param_idx:param_idx + hi], self.h[i-1]])
                #updates.append((self.wf[i], self.wf[i] - l_r*T.sum(wf_upd, axis=0)))
                theta_upd = T.set_subtensor(theta_upd[th_idx[0]:th_idx[1]], (self.wf[i] - l_r*T.sum(wf_upd, axis=0)).flatten())

            bf_upd = T.mean(g[:,:, param_idx:param_idx+hi], axis=1)
            #updates.append((self.bf[i], self.bf[i]- l_r*T.sum(bf_upd)))
            th_idx = self.theta_idx['bf[{0}]'.format(i)]
            theta_upd = T.set_subtensor(theta_upd[th_idx[0]:th_idx[1]], self.bf[i]- l_r*T.sum(bf_upd))
            param_idx += hi
        updates.append((self.theta, theta_upd))


        #self.tmp_g = theano.function(inputs=[self.x, self.y, self.m], outputs=[go, g, self.norm, self.delta_o])

        train_model = theano.function(inputs=[self.x, self.y, self.m],
            outputs=[self.c, self.a[0], self.ph[0], self.h[0], self.o, self.pyi, self.pyi_prime, self.py, self.y_pred, go, g, self.ph_prime[0]],
            updates=updates,
            givens={l_r: learning_rate})
        print "Compiled"
        for e in xrange(epochs):
            c = 0
            c, a, ph, h, o, pyi, pyi_prime, py, y_pred, gom, g, ph_prime  = train_model(x, y, m)
            print "Epoch {0} error: {1}".format(e,c)
#            print "Epoch {0}: error: {1}, a {2} ph {3} h {4} o {5} pyi {6} pyi_prime {7} py {8} y_pred {9} ".format(e, c, a[0,:4], ph[0,:4], h[0,:4], o[0,:4], pyi[0,:4], pyi_prime[0,:4], py[0,:4], y_pred[0,:4])

    def gradient_pyh(self):
        def single_go(o_prime, o, normalization):
            return -1.*o_prime/normalization, o*1./normalization 
        norm = T.sum(self.delta_o, axis=0)
        [g_om, w_norm], _ = theano.scan(single_go,  sequences=[self.delta_o_prime, self.delta_o], non_sequences=norm, outputs_info=[None, None])
        return -1.*g_om, w_norm

    def get_gradient_2(self):

        gom, _ = self.gradient_pyh()

        def get_g2(go):
            n_hidden_total = np.sum(self.n_hidden)
            g = T.zeros((self.x.shape[0], n_hidden_total))
            param_idx = 0
            for i, hi in enumerate(self.n_hidden[-1::-1]):
                if i == 0:
                    g = T.set_subtensor(g[:, n_hidden_total-hi:], T.dot(go, self.w_out))
                else:
                    g = T.set_subtensor(g[:, n_hidden_total-param_idx-hi:n_hidden_total-param_idx], T.dot(g[:, n_hidden_total-param_idx:n_hidden_total-param_idx+self.n_hidden[-i]], self.wf[self.n_hidden.size-i]))
                param_idx += hi
            return g
        g2, _ = theano.scan(get_g2, sequences=gom)

        return gom, g2

    def get_gradient_3(self):

        gom, _ = self.gradient_pyh()

        def get_g2(go, it):
            n_hidden_total = np.sum(self.n_hidden)
            g = T.zeros((self.x.shape[0], n_hidden_total))
            param_idx = 0

            for i, hi in enumerate(self.n_hidden[-1::-1]):
                if i == 0:
                    g = T.set_subtensor(g[:, n_hidden_total-hi:], self.ph_prime[i][it]*T.dot(go, self.w_out))
                else:
                    g = T.set_subtensor(g[:, n_hidden_total-param_idx-hi:n_hidden_total-param_idx], self.ph_prime[i][it]*T.dot(g[:, n_hidden_total-param_idx:n_hidden_total-param_idx+self.n_hidden[-i]], self.wf[self.n_hidden.size-i]))
                param_idx += hi
            return g

        g3, _ = theano.scan(get_g2, sequences=[gom, theano.tensor.arange(self.m)])
        return gom, g3

    def get_gradient_4(self):
        """
        w_mean = self.py*1./T.sum(self.py, axis=0)
        def get_phm(phi, h):
            return T.prod(h*phi + (1-h)*(1-phi), axis=1)

        logf = T.zeros(self.py.shape)
        for i in xrange(len(self.n_hidden)):
            phlm, _ = theano.scan(get_phm, sequences=[self.ph[i], self.h[i]])
            logf += T.log(phlm)
        
        logf += T.log(self.py)
        f = T.sum(w_mean*logf)
        return T.grad(f, self.theta)
        """
        g_om, w_norm = self.gradient_pyh()
        def single_go(a_prime, a, w_norm):
            return w_norm*a_prime*1./a

        n_hidden_total = np.sum(self.n_hidden)
        g4 = T.zeros((self.m, self.x.shape[0], n_hidden_total))
        param_idx = 0

        for i, hi in enumerate(self.n_hidden):
            g_aim,_ = theano.scan(single_go,  sequences=[self.delta_a_prime[i], self.delta_a[i], w_norm])
            g4 = T.set_subtensor(g4[:,:,param_idx:param_idx+hi], g_aim)
            param_idx += hi
        
        return g_om, g4, w_norm



    def get_gradient_5(self):
        """
        w_mean = self.py*1./T.sum(self.py)
        phm = (T.dot(self.h[0], self.ph[0].T) + T.dot(1-self.h[0], 1-self.ph[0].T)).flatten()
        logphm = T.log(phm)
        f = T.dot(w_mean, T.log(self.py) + logphm) - 1./m*T.sum(logphm)
        self.test = theano.function(inputs=[self.x, self.m], outputs=T.grad(f, self.theta))
        return T.grad(f, self.theta)
        """
        g_om, w_norm = self.gradient_pyh()
        def single_go(a_prime, a, w_norm):
            return w_norm*a_prime*1./a
        g5 = []
        for i in xrange(self.n_hidden.size):
            g_aim,_ = theano.scan(single_go,  sequences=[self.delta_a_prime[i], self.delta_a[i]], non_sequences=(w_norm-1./self.m))
            g5.append(g_aim)
        return g_om, g5



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

np.random.seed(0)
sfn = SFN(5,[4, 3],2,['sigmoid', 'sigmoid', 'sigmoid'])
x = np.array([[1.5, 0, -1, 5, 0.4], [-1,2,3,0, -2.4], [-1,2,3,0, -2.4]])
m = 5
gradient_type = 'g1'
learning_rate = 0.00005
epochs = 60
y =  np.array([[0, 1],[1,1], [0,0]])
#l = sfn.tmp(x,y,m)

#names = ["a0", "a1", "ph0", "ph1", "h0", "h1", "o", "pyi", "pyi_prime", "y pred", "py", "delta o", "delta o prime", "cm", "c"]

#for i, li in enumerate(l):
#    print "--{0}--".format(names[i])
#    print li
sfn.fit(x,y,m, 'g4', learning_rate, epochs)
print "HERE"
import cPickle, gzip, numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
x_train = train_set[0]
y_train = train_set[1]
y_val = valid_set[1]
f.close()

x_train = x_train[:10,:]
y_train = y_train[:10].reshape(-1,1)

y_train = y_train == 4
y_val = y_val == 4
#sfn = SFN(x_train.shape[1],[500], 1,['sigmoid', 'sigmoid'])
#names = ["a0", "ph0", "h0", "o", "pyi", "pyi_prime", "y pred", "py", "delta o", "delta o prime", "cm", "c"]
#l = sfn.tmp(x_train, y_train, m)
#for i, li in enumerate(l):
#    if i == len(l)-2:
#        break
#    print "--{0}--".format(names[i])
#    print li[0,:5]
#print sfn.tmp2(x_train,y_train,m)
#sfn.fit(x_train, y_train, m, gradient_type, learning_rate, epochs)

#go, g, norm, delta_o =  sfn.tmp_g(x_train, y_train, m)
#print go[0,:5]
#print g[0,:5]
#print norm[:5]
#print delta_o[:,:5]
"""
y_pred_train = -1*np.ones((y_train.shape[0], m))

for i in xrange(y_pred_train.shape[0]):
    y_pred_train[i] = sfn.predict(x_train[i].reshape(1,-1), m).flatten()

y_pred_val = -1*np.ones(y_val.shape[0], m)
for i in xrange(y_pred_val[0]):
    y_pred_val[i] = sfn.predict_((x_val[i].reshape(1,-1), m)).flatten()

print np.sum(y_train*y_pred_train)

print np.sum(y_val*y_pred_val)
import ipdb
ipdb.set_trace()
"""


