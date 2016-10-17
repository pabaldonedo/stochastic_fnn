import numpy as np
import theano
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from util import load_states
from util import load_controls
from util import log_init
from util import load_files
from util import flatten
from classifiers import MLPClassifier
from classifiers import RecurrentMLP
from classifiers import ResidualMLPClassifier
from classifiers import BoneResidualMLPClassifier
from classifiers import BoneMLPClassifier
from classifiers import Classifier


def load_3d(idx_path):
    sampled_clipped = False
    lagged = False
    n_out = 30

    idx_train_file = '{0}/idx_train.txt'.format(idx_path)
    idx_test_file = '{0}/idx_test.txt'.format(idx_path)
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

    idx_train = np.genfromtxt(idx_train_file, delimiter=',', dtype=int)
    idx_test = np.genfromtxt(idx_test_file, delimiter=',', dtype=int)
    y_test = y[idx_test]
    y_train = y[idx_train]

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
    x_test = x[idx_test]
    x_train = x[idx_train]
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    return x_test, y_test, n_train, n_test, x_train, y_train


def load_2d(idx_path):
    controls_file = '2d/data/merged_controls.txt'
    states_file = '2d/data/merged_starting_states.txt'

    x_info_file = '2d/mux_stdx.csv'
    y_info_file = '2d/muy_stdy.csv'

    idx_train_file = '{0}/idx_train.txt'.format(idx_path)
    idx_test_file = '{0}/idx_test.txt'.format(idx_path)

    x_info = np.asarray(np.genfromtxt(
        x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(
        y_info_file, delimiter=','), dtype=theano.config.floatX)
    print "means loaded"

    y = np.asarray(pd.read_csv(
        controls_file, delimiter=',',
        header=None).values, dtype=theano.config.floatX)

    muy = y_info[0]
    stdy = y_info[1]
    stdy[stdy == 0] = 1.
    y = (y - muy) * 1. / stdy
    idx_train = np.genfromtxt(idx_train_file, delimiter=',', dtype=int)
    idx_test = np.genfromtxt(idx_test_file, delimiter=',', dtype=int)

    y_train = y[idx_train]
    y_test = y[idx_test]

    x = np.asarray(pd.read_csv(
        states_file, delimiter=',',
        header=None).values, dtype=theano.config.floatX)

    mux = np.mean(x, axis=0)
    stdx = np.std(x, axis=0)
    stdx[stdx == 0] = 1.

    x = (x - mux) * 1. / stdx
    x_train = x[idx_train]
    x_test = x[idx_test]

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    return x_test, y_test, n_train, n_test, x_train, y_train


def main():

    folders =  ["residual_mlp_classifier_n_16_n_impulse_2000_0_mlp_n_hidden_[[150, 100]]_mlp_activation_[['tanh', 'tanh']]_bsize_100_method_SGD_bn_False_dropout_False_lagged_False"]
    nnames =   ['residual_mlp_classifier_n_hidden_[[150, 100]]']

    network_names = ['network_output/{0}/networks/{1}'.format(
        folder, nname) for folder, nname in zip(folders, nnames)]
    network_type = 'residual'
    network_types = ['mlp', 'residual', 'bone_residual', 'bone_mlp', 'lbn']
    assert network_type in network_types

    opath = [
        'network_output/{0}/likelihoods/'.format(folder) for folder in folders]
    ofile_l = ['{0}_test.csv'.format(nname) for nname in nnames]
    ofile_n = 'norms.csv'

    epoch0s = [10]  # *len(folders)
    epochns = [460]
    epoch_steps = [10] * len(folders)
    n_out = 30
    m = 1

    if network_type != 'lbn':
        assert m == 1

    for i in xrange(len(folders)):

        epoch0 = epoch0s[i]
        epochn = epochns[i]
        epoch_step = epoch_steps[i]
        opath = 'network_output/{0}/likelihoods/'.format(folders[i])
        ofile_l = '{0}_test.csv'.format(nnames[i])
        ofile_n = 'norms.csv'
        network_name = network_names[i]
        train_log_likelihood = np.genfromtxt(
            '{0}{1}.csv'.format(opath, nnames[i]), delimiter=',')
        norm_evolution = []
        log_likelihood = []

        x_test, y_test, n_train, n_test, x_train, y_train =         load_3d(
            'network_output/{0}'.format(folders[i]))
#        load_2d(
        #    'network_output/{0}'.format(folders[i]))


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
            elif network_type is network_types[3]:
                c = BoneMLPClassifier.init_from_file(
                    '{0}_epoch_{1}.json'.format(
                        network_name, epoch))
            elif network_type is network_types[4]:
                c = Classifier.init_from_file(
                    '{0}_epoch_{1}.json'.format(
                        network_name, epoch))

            likelihood_precision = c.likelihood_precision

            norms = [1. / p.size * (p**2).sum() for p in flatten(c.params)]
            #if network_type is network_types[4]:
                #compute_error_and_norms = theano.function(inputs=[c.x, c.y, c.m], outputs=[c.get_cost(0,0)]+norms)
                # cost_and_norm = compute_error_and_norms(
                 #   x_test, y_test[:, :n_out], m)
            if network_type is network_types[4]:
                compute_error = theano.function(
                inputs=[c.x, c.y, c.m], outputs=c.get_cost(0, 0))
                cost = compute_error(x_test, y_test[:, :n_out], m)

            else:
                compute_error = theano.function(
                                    inputs=[c.x,c.y], outputs=c.get_cost(0,0))
                cost = compute_error(x_test, y_test[:,:n_out])
            """
            else:
                compute_error_and_norms = theano.function(
                    inputs=[c.x, c.y], outputs=[c.get_cost(0, 0)] + norms)
                cost_and_norm = compute_error_and_norms(
                    x_test, y_test[:, :n_out])

                norms = cost_and_norm[1:]
                norm_evolution.append(norms)
                cost = cost_and_norm[0]

                norms = np.array(norm_evolution)

                fig, ax = plt.subplots()
                for i in xrange(norms.shape[1]):
                    ax.plot(np.arange(epoch0, epochn +
                                      1, epoch_step), norms[:, i])
                ax.set_xlabel('epoch')
                ax.set_ylabel('norm')
                legend = ['{0}{1}'.format(str(p), i // 2)
                          for i, p in enumerate(flatten(c.params))]

                ax.legend(legend)

                plt.savefig(opath + 'norms.png')
                np.savetxt(opath + ofile_n, np.hstack((
                    np.arange(epoch0, epochn + 1, epoch_step).reshape(-1, 1), norms)), delimiter=',')
            """
            log_likelihood.append(-1 * cost * x_test.shape[0])
            print cost

        log_l = np.array(log_likelihood).reshape(-1, 1)
        normed_log_l = log_l - n_test * n_out / 2. * \
            np.log(2 * np.pi * 1. / likelihood_precision) - n_test * np.log(m)
        print 1. / n_test * normed_log_l

        fig, ax = plt.subplots()
        ax.plot(train_log_likelihood[:, 0], 1. /
                n_train * train_log_likelihood[:, 1], 'b')
        ax.plot(np.arange(epoch0, epochn + 1, epoch_step),
                1. / n_test * normed_log_l, 'r')
        ax.set_xlabel('epoch')
        ax.set_ylabel('log likelihood')
        ax.legend(['Train', 'Test'])
        plt.savefig(opath + 'log_likelihood.png')

        np.savetxt(opath + ofile_l, np.hstack((
            np.arange(epoch0, epochn + 1, epoch_step).reshape(-1, 1), log_l)), delimiter=',')


if __name__ == '__main__':
    main()
