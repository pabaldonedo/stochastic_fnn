"""
Handcrafted LBN that models a bimodal normal distribution.
"""

from lbn import LBN
import theano
import theano.tensor as T
import numpy as np
from util import flatten
import matplotlib.pyplot as plt
from multimodal_experiment import MultimodalGenerator
import scipy.stats as stats

mgen = MultimodalGenerator()
x, y = mgen.generate_classes(n_classes=1, n=200)


n_in = 1
lbn_n_hidden = [1]
n_out = 1
det_activations = ['linear', 'linear']
stoch_activations = ['sigmoid', 'sigmoid']

lbn = LBN(n_in, lbn_n_hidden,
                       n_out,
                       det_activations,
                       stoch_activations)

mu1 = mgen.mu_extreme[0]
mu2 = mgen.mu_extreme[1]
std = mgen.std_extreme

#Det W
lbn.params[0][0].set_value(np.asarray(np.array(1).reshape(1,1), dtype=theano.config.floatX))
#Det b
lbn.params[0][1].set_value(np.asarray(np.array(1).reshape(1,), dtype=theano.config.floatX))

#Stochastic
lbn.params[0][2].set_value(np.asarray(np.array(1).reshape(1,1), dtype=theano.config.floatX))
lbn.params[0][3].set_value(np.asarray(np.array(-1).reshape(1,), dtype=theano.config.floatX))
lbn.params[0][4].set_value(np.asarray(np.array(2).reshape(1,1), dtype=theano.config.floatX))
lbn.params[0][5].set_value(np.asarray(np.array(-1).reshape(1,), dtype=theano.config.floatX))

#Output layer
lbn.params[1][0].set_value(np.asarray(np.array(mu2-mu1).reshape(1,1), dtype=theano.config.floatX))
lbn.params[1][1].set_value(np.asarray(np.array(mu1).reshape(1,), dtype=theano.config.floatX))

n = 4000
output = np.zeros(n)
for i in xrange(n):
    mu = lbn.predict(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX),1)
    output[i] = np.random.normal(loc=mu, scale=std)

x_axis = np.linspace(0, 255, 100)
normal1, _, _ = mgen.get_true_distributions(x_axis)

plt.plot(x_axis, normal1)

plt.hist(output, bins=100, normed=True)
plt.show()

def check_model(lbn2, normed=False, std=1):
    activated = theano.function(inputs=[lbn2.x, lbn2.m], outputs=[lbn2.output, lbn2.hidden_layers[0].stoch_layer.ph])

    n2 = 1000
    mus2 = np.zeros(n2)
    output2 = np.zeros(n2)
    for i in xrange(n2):
        mus2[i], h = activated(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX),1)
        #print h
        output2[i] = np.random.normal(loc=mus2[i], scale=std)

    x_axis = np.linspace(-2, 2, 100)
    plt.plot(x_axis, 0.5*stats.norm.pdf(x_axis, -1, 0.07)+0.5*stats.norm.pdf(x_axis2, 1, 0.07))
    plt.hist(output2, bins=100, normed=True)
    plt.show()

  
#Learned LBN
x, y = mgen.generate_classes(n_classes=1, n=200)
y_transform = (y-np.mean(y, axis=0))*1./(np.std(y, axis=0))

#x_axis2 = np.linspace(-2, 2, 100)
#plt.plot(x_axis2, 0.5*stats.norm.pdf(x_axis2, (mgen.mu_extreme[0]-np.mean(y, axis=0)[0])*1./np.std(y, axis=0), mgen.std_extreme*1./np.std(y, axis=0))+0.5*stats.norm.pdf(x_axis2, (mgen.mu_extreme[1]-np.mean(y, axis=0)[0])*1./np.std(y, axis=0), mgen.std_extreme*1./np.std(y, axis=0)))
#plt.hist(y_transform, bins=100, normed=True)
#plt.show()


lbn2 = LBN(n_in, lbn_n_hidden,
                       n_out,
                       det_activations,
                       stoch_activations)

print "RANDOMNESS"
for p in flatten(lbn2.params):
    print p.get_value()


lbn2.params[0][1].set_value(np.asarray(np.array(-2).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][2].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][3].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][4].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][5].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))

lbn2.params[1][1].set_value(np.asarray(np.array(.4).reshape(1,), dtype=theano.config.floatX))

print "HANDCRAFTED"
for p in flatten(lbn2.params):
    print p.get_value()


m = 10

sigma = .1
exp_arg = -0.5/sigma**2*(lbn2.y-lbn2.output)**2
exp = T.exp(exp_arg)
cost = -T.sum(T.log(T.sum(exp, axis=0))+T.log(1./lbn2.m *1./(T.sqrt(2*np.pi)*sigma)))
upds = []

lr = 1e-4
#for p in flatten(lbn2.params):
#    upds += [(p, p-lr*T.grad(cost, p))]
upds += [(lbn2.params[0][0], lbn2.params[0][0]-lr*T.grad(cost,lbn2.params[0][0]))]
upds += [(lbn2.params[0][1], lbn2.params[0][1]-lr*T.grad(cost,lbn2.params[0][1]))]

upds += [(lbn2.params[1][0], lbn2.params[1][0]-lr*T.grad(cost,lbn2.params[1][0]))]
upds += [(lbn2.params[1][1], lbn2.params[1][1]-lr*T.grad(cost,lbn2.params[1][1]))]

train_model = theano.function(inputs=[lbn2.x, lbn2.y, lbn2.m], outputs=[cost, exp], updates=upds)
get_exp_cost = theano.function(inputs=[lbn2.x, lbn2.y, lbn2.m], outputs=[lbn2.output, exp_arg, exp, cost])

print "TRAINING"
print "EPOCH 0---"
o0, exp_arg0, exp0, cost0 = get_exp_cost(x, y_transform, m)

def check_true_predict_p(mus, y, std=0.1):


    return 0.5*stats.norm.pdf(y, mus[0], std)+0.5*stats.norm.pdf(y, mus[1], std)




#stats.norm.pdf(1, loc=0, scale=1)
print np.unique(o0)
#print exp_arg0
#print exp0
print cost0
print "---"
epochs = 100
errors = np.zeros(epochs)
for e in xrange(epochs):

    errors[e], exp_train = train_model(x, y_transform, m)

    print errors[e]
    o2, exp_arg2, exp2, cost2 = get_exp_cost(x, y_transform, m)
    print np.unique(o2)


x_axis2=np.linspace(-2,2,100)

plt.plot(x_axis2, 0.5*stats.norm.pdf(x_axis2, mus2[0], sigma)+0.5*stats.norm.pdf(x_axis2, mus2[1], sigma))
plt.show()


np.savetxt('/Users/Pablo/Dropbox/aalto/thesis/images/multimodal_handcrafted_y.csv', y, delimiter=',')
np.savetxt('/Users/Pablo/Dropbox/aalto/thesis/images/multimodal_handcrafted_y_transform.csv', y_transform, delimiter=',')

"""COnverged:
used m = 10000
n_samples = 10000

sigma = 1
means [-0.4241682   0.40020588]

params:
[[ 0.9150387]]
[-1.45311594]
[[ 0.]]
[ 0.]
[[ 0.]]
[ 0.]
[[-0.56731474]]
[-0.4241682]

lbn2.params[0][0].set_value(np.asarray(np.array(0.9150387).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][1].set_value(np.asarray(np.array(-1.45311594).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][2].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][3].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][4].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][5].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))
lbn2.params[1][0].set_value(np.asarray(np.array(-0.56731474).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[1][1].set_value(np.asarray(np.array(-0.4241682).reshape(1,), dtype=theano.config.floatX))


Convreged
sigma = 0.5
means [-1.01164603  0.9821558 ]

parmas:
[[ 0.9150387]]
[-1.7485528]
[[ 0.]]
[ 0.]
[[ 0.]]
[ 0.]
[[-1.14025831]]
[-1.01164603]

lbn2.params[0][0].set_value(np.asarray(np.array(0.9150387).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][1].set_value(np.asarray(np.array(-1.7485528).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][2].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][3].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))
lbn2.params[0][4].set_value(np.asarray(np.array(0).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[0][5].set_value(np.asarray(np.array(0).reshape(1,), dtype=theano.config.floatX))
lbn2.params[1][0].set_value(np.asarray(np.array(-1.14025831).reshape(1,1), dtype=theano.config.floatX))
lbn2.params[1][1].set_value(np.asarray(np.array(-1.01164603).reshape(1,), dtype=theano.config.floatX))


"""