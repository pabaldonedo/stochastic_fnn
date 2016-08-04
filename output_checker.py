from util import load_states
from util import load_controls
from classifiers import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import theano
import os

n=1
x = load_states(n)
y = load_controls(n)

x_info_file = 'mux_stdx_n_16_n_impulse_2000_5.csv'#'sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv'
y_info_file = 'muy_stdy_n_16_n_impulse_2000_5.csv'#'sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv'

x_info = np.asarray(np.genfromtxt(x_info_file, delimiter=','), dtype=theano.config.floatX)
y_info = np.asarray(np.genfromtxt(y_info_file, delimiter=','), dtype=theano.config.floatX)

mux = x_info[0]
stdx = x_info[1]

muy = y_info[0]
stdy = y_info[1]

network_id = 'mlp_classifier_n_16_n_impulse_2000_0_mlp_n_hidden_[150]_mlp_activation_[sigmoid]_bsize_100_method_SGD_bn_False'
fname = 'network_output/{0}/networks/mlp_n_hidden_[150]_epoch_1000.json'.format(network_id)
c = MLPClassifier.init_from_file(fname)
cols = [1] + list(range(3, x.shape[1]))
y_hat = np.zeros((y.shape[0], 30))
for i, xi in enumerate(x):
    yi = c.predict((xi[cols].reshape(1,-1)-mux[cols])*1./stdx[cols])*stdy[:30]+muy[:30]
    y_hat[i] = yi

fpath = 'output_plot/{0}'.format(network_id)
if not os.path.exists(fpath):
        os.makedirs(fpath)
for i in xrange(x.shape[0]*1/61):
    plt.figure()
    fig, ax = plt.subplots(6,5,figsize=(20, 20))
    ax = ax.flatten()
    for j in xrange(30):
        ax[j].plot(y_hat[i*61:61*(i+1),j], color='r')
        ax[j].plot(y[i*61:61*(i+1), j], color='b')
    plt.savefig('{0}/{1}.pdf'.format(fpath,i))

