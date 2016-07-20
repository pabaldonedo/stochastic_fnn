import numpy as np
import theano
import os
import warnings
import pandas as pd
from util import load_states
from util import load_controls
from util import log_init
from util import load_files
from classifiers import MLPClassifier
from classifiers import RecurrentMLP
from classifiers import ResidualMLPClassifier
from classifiers import BoneResidualMLPClassifier


def main():
    network_name = 'network_output/Testr/networks/residual_mlp_classifier_n_hidden_[[150, 150],[100, 100],[80, 80],[50, 50]]'
    network_type = 'residual'
    network_types = ['mlp', 'residual', 'bone_residual']
    assert network_type in network_types
    sampled_clipped = False
    lagged = False
    n_out = 30

    ofile = 'network_output/Testr/likelihoods/residual_mlp_classifier_n_hidden_[[150, 150],[100, 100],[80, 80],[50, 50]]_test.csv'

    idx_file = 'network_output/Testr/prueba_test.txt'
#    x_info_file = 'mux_stdx_lagged_n_16.csv'
#    x_info_file = 'sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv'
    x_info_file = 'mux_stdx_n_16_n_impulse_2000_5.csv'
#    y_info_file = 'sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv'
    y_info_file = 'muy_stdy_n_16_n_impulse_2000_5.csv'

    # mean and std files:
    x_info = np.asarray(np.genfromtxt(
        x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(
        y_info_file, delimiter=','), dtype=theano.config.floatX)
    print "means loaded"

    n = 16
    n_impulse_2000 = 0

    epoch0 = 640
    epochn = 1570
    epoch_step = 10

    if sampled_clipped:
        y = np.asarray(pd.read_csv(
            'data/sample_clipped_controls_n_16_n_impulse_2000_5.txt', delimiter=',',
            header=None).values, dtype=theano.config.floatX)
    else:
        y = load_controls(n)
        if n_impulse_2000 > 0:
            y_impulse = load_files(n_impulse_2000, 'controls_impulse_2000')
            y = np.vstack((y, y_impulse))
    muy = y_info[0]
    stdy = y_info[1]
    y = (y - muy) * 1. / stdy

    idx = np.asarray(np.genfromtxt(idx_file, delimiter=','), dtype='int')
    y_test = y[idx]

    if sampled_clipped:
        x = np.asarray(pd.read_csv(
            'data/sample_clipped_states_n_16_n_impulse_2000_5.txt', delimiter=',', header=None).values, dtype=theano.config.floatX)
    elif lagged:
        x = load_files(n, 'lagged_states')
        assert not n_impulse_2000 > 0
    else:
        x = load_states(n)
        if n_impulse_2000 > 0:
            x_impulse = load_files(n_impulse_2000, 'states_impulse_2000')
            x = np.vstack((x, x_impulse))

    mux = x_info[0]
    stdx = x_info[1]
    x = (x - mux) * 1. / stdx
    if lagged:
        cols = [1] + list(range(3, 197)) + [198] + \
            list(range(200, x.shape[1]))
    else:
        cols = [1] + list(range(3, x.shape[1]))

    x = x[:, cols]
    x_test = x[idx]

    log_likelihood = []

    for epoch in xrange(epoch0, epochn + 1, epoch_step):

        if network_type is network_types[0]:
            c = MLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    network_name,  epoch))
        elif network_type is network_types[1]:
            c = ResidualMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    network_name,  epoch))
        elif network_type is network_types[2]:
            c = ResidualMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    network_name,  epoch))

        cost = c.compute_error(
            x_test, y_test[:, :n_out])
        print cost
        log_likelihood.append(cost * x_test.shape[0])

    log_l = np.array(log_likelihood).reshape(-1, 1)
    np.savetxt(ofile, np.hstack((
        np.arange(epoch0, epochn + 1, epoch_step).reshape(-1, 1), log_l)), delimiter=',')


if __name__ == '__main__':
    main()
