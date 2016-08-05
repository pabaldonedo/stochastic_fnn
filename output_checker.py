from util import load_states
from util import load_controls
from classifiers import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import theano
import os

def load_data_3d(n=1):
    x = load_states(n)
    y = load_controls(n)
    cols = [1] + list(range(3, x.shape[1]))

    x_info_file = 'mux_stdx_n_16_n_impulse_2000_5.csv'#'sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv'
    y_info_file = 'muy_stdy_n_16_n_impulse_2000_5.csv'#'sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv'

    x_info = np.asarray(np.genfromtxt(x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(y_info_file, delimiter=','), dtype=theano.config.floatX)

    mux = x_info[0][cols]
    stdx = x_info[1][cols]

    muy = y_info[0][:30]
    stdy = y_info[1][:30]

    return x[:,cols], y, mux, stdx, muy, stdy

def load_data_2d(n=1e5):

    y = np.asarray(pd.read_csv("2d/data/merged_controls.txt",
                               delimiter=',', header=None).values, dtype=theano.config.floatX)

    x = np.asarray(pd.read_csv("2d/data/merged_starting_states.txt",
                               delimiter=',', header=None).values, dtype=theano.config.floatX)

    x_info_file = '2d/mux_stdx.csv'
    y_info_file = '2d/muy_stdy.csv'

    x_info = np.asarray(np.genfromtxt(x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(y_info_file, delimiter=','), dtype=theano.config.floatX)


    mux = x_info[0]
    stdx = x_info[1]

    muy = y_info[0]
    stdy = y_info[1]

    return x[:n], y[:n], mux, stdx, muy, stdy


def plot_3d(x, y, y_hat):

    for i in xrange(x.shape[0]*1/61):
        plt.figure()
        fig, ax = plt.subplots(6,5,figsize=(20, 20))
        ax = ax.flatten()
        for j in xrange(30):
            ax[j].plot(y_hat[i*61:61*(i+1),j], color='r')
            ax[j].plot(y[i*61:61*(i+1), j], color='b')
        plt.savefig('{0}/{1}.pdf'.format(fpath,i))

def plot_2d(x, y, y_hat):

    for i in xrange(x.shape[0]*1/61):
        plt.figure()
        fig, ax = plt.subplots(1,3,figsize=(30, 10))
        ax = ax.flatten()
        for j in xrange(3):
            ax[j].plot(y_hat[i*61:61*(i+1),j], color='r')
            ax[j].plot(y[i*61:61*(i+1), j], color='b')
        plt.savefig('{0}/{1}.pdf'.format(fpath,i))



x, y, mux, stdx, muy, stdy = load_data_2d()

network_id = 'mlp_classifier_mlp_n_hidden_[20,15]_mlp_activation_[relu,relu]_bsize_100_method_SGD_bn_False_dropout_False'
fname = '2d/network_output/{0}/networks/mlp_classifier_n_hidden_[20,15]_epoch_250.json'.format(network_id)
fpath = '2d/output_plot/{0}'.format(network_id)

c = MLPClassifier.init_from_file(fname)

y_hat = np.zeros((y.shape[0], y.shape[1]))
for i, xi in enumerate(x):
    yi = c.predict((xi.reshape(1,-1)-mux)*1./stdx)*stdy+muy
    y_hat[i] = yi

if not os.path.exists(fpath):
    os.makedirs(fpath)
plot_2d(x, y, y_hat)
