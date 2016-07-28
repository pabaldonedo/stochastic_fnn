import numpy as np
import theano
import os
import warnings
import pandas as pd
from util import load_states
from util import load_controls
from util import log_init
from util import load_files
from classifiers import RecurrentClassifier
from classifiers import Classifier
from classifiers import ResidualClassifier


def main():

    network_type = 'classifier'
    load_idx = False
    idx_train_file = None
    idx_test_file = None

    assert not (load_idx and idx_train_file is None)

    network_types = ['classifier', 'residual']

    assert network_type in network_types

    load_means_from_file = True

    data_x_from_file_name = None
    data_y_from_file_name = 'clipped_controls_n_16_n_impulse_2000_5.txt'
    extra_name_tag = 'clipped'

    lagged = False
    if data_x_from_file_name is not None or data_y_from_file_name is not None:
        print "WARNING DATA FROM FILE"

    # x_info_file = 'sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv'
    x_info_file = 'mux_stdx_n_16_n_impulse_2000_5.csv'
    #_info_file = 'sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv'
    y_info_file = 'muy_stdy_n_16_n_impulse_2000_5.csv'

    # mean and std files:
    x_info = np.asarray(np.genfromtxt(
        'mux_stdx_n_16_n_impulse_2000_5.csv', delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(
        'muy_stdy_n_16_n_impulse_2000_5.csv', delimiter=','), dtype=theano.config.floatX)

    print "means loaded"
    assert not (load_means_from_file and x_info is None and y_info is None)
    # Number of datasets
    n = 16
    n_impulse_2000 = 5

    # RNN on top of LBN
    recurrent = False

    # Only for NO recurrent
    feet_learning = False
    feet_min = 50
    feet_max = 100

    assert not (
        feet_learning and recurrent), "Feet learning and recurrent cannot be true at the same time"

    # Load data
    seq_len = 61

    if data_y_from_file_name is not None:
        y = np.asarray(pd.read_csv(
            'data/{0}'.format(data_y_from_file_name), delimiter=',',
            header=None).values, dtype=theano.config.floatX)
    else:
        y = load_controls(n)
        if n_impulse_2000 > 0:
            y_impulse = load_files(n_impulse_2000, 'controls_impulse_2000')
            y = np.vstack((y, y_impulse))

    if load_means_from_file:
        muy = y_info[0]
        stdy = y_info[1]
    else:
        muy = np.mean(y, axis=0)
        stdy = np.std(y, axis=0)
    stdy[stdy == 0] = 1.

    if feet_learning:
        feet_idx = np.arange(y.shape[0])
        feet_idx = feet_idx[np.any(np.logical_and(
            np.abs(y[:, 6:16]) >= feet_min, np.abs(y[:, 6:16]) < feet_max), axis=1)]
        y = y[feet_idx, :]

    train_size = 0.8

    print "controls loaded"
    y = (y - muy) * 1. / stdy
    if recurrent:
        y = y.reshape(seq_len, -1, y.shape[1])
        if not load_idx:
            idx = np.random.permutation(y.shape[1])

            y = y[:, idx, :-4]
            train_bucket = int(np.ceil(y.shape[1] * train_size))
            y_train = y[:, :train_bucket]
            y_test = y[:, train_bucket:]
        else:
            idx_train = np.genfromtxt(idx_train_file, delimiter=',', dtype=int)
            idx_test = np.genfromtxt(idx_test_file, delimiter=',', dtype=int)
            y_train = y[:, idx_train, :-4]
            y_test = y[:, idx_test, :-4]

    else:
        if not load_idx:
            idx = np.random.permutation(y.shape[0])
            y = y[idx, :-4]
            train_bucket = int(np.ceil(y.shape[0] * train_size))
            y_train = y[:train_bucket]
            y_test = y[train_bucket:]
        else:
            idx_train = np.genfromtxt(idx_train_file, delimiter=',', dtype=int)
            idx_test = np.genfromtxt(idx_test_file, delimiter=',', dtype=int)
            y_train = y[idx_train, :-4]
            y_test = y[idx_test, :-4]

    if data_x_from_file_name is not None:
        x = np.asarray(pd.read_csv(
            'data/{0}'.format(data_x_from_file_name), delimiter=',', header=None).values, dtype=theano.config.floatX)
    elif lagged:
        x = load_files(n, 'lagged_states')
        assert not n_impulse_2000 > 0
    else:
        x = load_states(n)
        if n_impulse_2000 > 0:
            x_impulse = load_files(n_impulse_2000, 'states_impulse_2000')
            x = np.vstack((x, x_impulse))

    if load_means_from_file:
        mux = x_info[0]
        stdx = x_info[1]
        if lagged:
            mux = np.hstack((mux, mux))
            stdx = np.hstack((stdx, stdx))
    else:
        mux = np.mean(x, axis=0)
        stdx = np.std(x, axis=0)

    stdx[stdx == 0] = 1.

    if feet_learning:
        x = x[feet_idx, :]
    print "states loaded"
    x = (x - mux) * 1. / stdx
    if recurrent:
        x = x.reshape(seq_len, -1, x.shape[1])
        if lagged:
            cols = [1] + list(range(3, 197)) + [198] +\
                list(range(200, x.shape[2]))
        else:
            cols = [1] + list(range(3, x.shape[2]))

        x = x[:, :, cols]
        if not load_idx:

            x = x[:, idx, :]

            x_train = x[:, :train_bucket]
            x_test = x[:, train_bucket:]
        else:
            x_train = x[:, idx[0]]
            x_test = x[:, idx[1]]

        n_in = x.shape[2]
        n_out = y.shape[2]
    else:

        if lagged:
            cols = [1] + list(range(3, 197)) + [198] +\
                list(range(200, x.shape[1]))
        else:
            cols = [1] + list(range(3, x.shape[1]))

        x = x[:, cols]

        if not load_idx:
            x = x[idx]
            x_train = x[:train_bucket]
            x_test = x[train_bucket:]

        else:
            x_train = x[idx_train]
            x_test = x[idx_test]

        n_in = x.shape[1]
        n_out = y.shape[1]

    print "Data ready to go"
    # MLP definition
    mlp_activation_names = ['sigmoid']
    mlp_n_in = 13
    mlp_n_hidden = [10]

    bone_networks = True
    # LBN definition
    lbn_n_hidden = [150]  # , 100, 50]
    det_activations = ['linear', 'linear']   # , 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    likelihood_precision = .1
    m = 10

    # RNN definiton + LBN n_out if RNN is the final layer
    rnn_type = "LSTM"
    rnn_hidden = [30]
    rnn_activations = [['sigmoid', 'tanh', 'sigmoid',
                        'sigmoid', 'tanh'], 'linear']  # ['sigmoid', 'linear']
    lbn_n_out = 50
    noise_type = 'multiplicative'

    # Fit options
    b_size = 100
    epoch0 = 1
    n_epochs = 10000
    lr = 1.
    save_every = 10  # Log saving
    chunk_size = 10000  # Memory chunks
    batch_normalization = False  # TODO FOR RECURRENT CLASSIFIER!
    dropout = False

    # Optimizer
    opt_type = 'SGD'
    method = {'type': opt_type, 'lr_decay_schedule': 'constant',
              'lr_decay_parameters': [lr],
              'momentum_type': 'nesterov', 'momentum': 0.01, 'b1': 0.9,
              'b2': 0.999, 'epsilon': 1e-8, 'rho': 0.99, 'e': 1e-8,
              'learning_rate': lr, 'dropout': dropout}

    # Load from file?
    load_from_file = False
    session_name = None
    load_different_file = False

    assert not (load_different_file and not load_from_file), "You have set load different_file to True but you are not loading any network!"

    # Saving options
    if recurrent:
        network_name = "{0}_n_{1}_n_impulse_2000_{2}_mlp_n_hidden_[{3}]_mlp_activation_[{4}]"\
            "_lbn_n_hidden[{5}]_det_activations_[{6}]_stoch"\
            "_activations_[{7}]_m_{8}_noise_type_{9}_bsize_{10}"\
            "_method_{11}_bn_{12}_dropout_{13}_lagged_{14}_{15}".\
            format(
                'recurrentclassifer_{0}'.format(rnn_type),
                n, n_impulse_2000,
                ','.join(str(e) for e in mlp_n_hidden),
                       ','.join(str(e) for e in mlp_activation_names),
                       ','.join(str(e) for e in lbn_n_hidden),
                       ','.join(str(e) for e in det_activations),
                       ','.join(str(e) for e in stoch_activations),
                       m, noise_type, b_size, method['type'],
                       batch_normalization, dropout,
                       lagged, extra_name_tag)

    else:
        if network_type == network_types[0]:
            network_name = "{0}_n_{1}_n_impulse_2000_{2}_mlp_n_hidden_[{3}]_mlp_activation_[{4}]"\
                "_lbn_n_hidden[{5}]_det_activations_[{6}]_stoch"\
                "_activations_[{7}]_m_{8}_noise_type_{9}_bsize_{10}"\
                "_method_{11}_bn_{12}_dropout_{13}_lagged_{14}_{15}".\
                format(
                    'residualclassifer' if network_type == network_types[
                        1] else 'classifier',
                    n, n_impulse_2000,
                    ','.join(str(e) for e in mlp_n_hidden),
                    ','.join(str(e) for e in mlp_activation_names),
                    ','.join(str(e) for e in lbn_n_hidden),
                    ','.join(str(e) for e in det_activations),
                    ','.join(str(e) for e in stoch_activations),
                    m, noise_type, b_size, method['type'],
                    batch_normalization, dropout,
                    lagged, extra_name_tag)

        elif network_type == network_types[1]:
            network_name = "{0}_n_{1}_n_impulse_2000_{2}"\
                "_hidden[{3}]_det_[{4}]_stoch"\
                "_[{5}]_m_{6}_{7}_bsize_{8}"\
                "{9}_bn_{10}_dropout_{11}_lagged_{12}_{13}".\
                format(
                    'residualclassifer' if network_type == network_types[
                        1] else 'classifier',
                    n, n_impulse_2000,
                    ','.join(str(e) for e in lbn_n_hidden),
                    ','.join(str(e) for e in det_activations),
                    ','.join(str(e) for e in stoch_activations),
                    m, noise_type, b_size, method['type'],
                    batch_normalization, dropout,
                    lagged, extra_name_tag)

    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)

    print "Paths created"
    fname = '{0}/{1}_lbn_n_hidden_[{2}]'.format(opath,
                                                'recurrentclassifier_{0}'.format(rnn_type) if recurrent
                                                else 'residualclassifier' if
                                                network_type == network_types[1] else
                                                'classifier',
                                                ','.join(str(e) for e in lbn_n_hidden))

    loaded_network_fname = '{0}/networks/{1}_lbn_n_hidden_[{2}]'.format(opath,
                                                                        'recurrentclassifier_{0}'.format(rnn_type) if recurrent
                                                                        else 'classifier' if
                                                                        network_type == network_types[1] else
                                                                        'residualclassifier',
                                                                        ','.join(str(e) for e in lbn_n_hidden))
    if load_different_file:
        warnings.warn(
            "CAUTION: loading log and network from different path than the saving path")

        loaded_network_folder = "{0}_n_13_n_impulse_2000_0_mlp_hidden_[{3}]_mlp_activation_[{4}]"\
            "_lbn_n_hidden_[{5}]_det_activations_[{6}]_stoch"\
            "_activations_[{7}]_m_{8}_noise_type_{9}_bsize_{10}"\
            "_method_{11}_bn_{12}".\
            format(
                'recurrentclassifier_{0}'.format(rnn_type) if recurrent
                else 'classifier',
                n, n_impulse_2000,
                ','.join(str(e) for e in mlp_n_hidden),
                ','.join(str(e) for e in mlp_activation_names),
                ','.join(str(e) for e in lbn_n_hidden),
                ','.join(str(e) for e in det_activations),
                ','.join(str(e) for e in stoch_activations),
                m, noise_type, b_size, method['type'], batch_normalization)
        loaded_opath = "network_output/{0}".format(loaded_network_folder)
        assert os.path.exists(
            loaded_opath), "Trying to load a network for non existing path: {0}".format(loaded_opath)

        loaded_network_name = "{0}_n_{1}_n_impulse_2000_{2}_mlp_hidden_[{3}]_mlp_activation_[{4}]"\
            "_lbn_n_hidden_[{5}]_det_activations_[{6}]_stoch"\
            "_activations_[{7}]_m_{8}_noise_type_{9}_bsize_{10}"\
            "_method_{11}".\
            format(
                'recurrentclassifier_{0}'.format(rnn_type) if recurrent
                else 'classifier',
                n, n_impulse_2000,
                ','.join(str(e) for e in mlp_n_hidden),
                ','.join(str(e) for e in mlp_activation_names),
                ','.join(str(e) for e in lbn_n_hidden),
                ','.join(str(e) for e in det_activations),
                ','.join(str(e) for e in stoch_activations),
                m, noise_type, b_size, method['type'], batch_normalization)  # "classifier_lbn_n_hidden_[150]"

        loaded_network_fname = '{0}/networks/{1}'.format(
            loaded_opath, loaded_network_name)

    else:
        loaded_opath = opath
    # LOGGING
    log, session_name = log_init(
        opath, session_name=session_name if load_from_file else None)

    print "LOG generated"
    if load_means_from_file:
        log.info('Loading means from x: {0}\ny: {1}'.format(
            x_info_file, y_info_file))
    else:
        log.info('Means and stds from data')

    if feet_learning:
        log.warning("Using feet learning.\nFeet min: {0}\nFeet max: {1}".format(
            feet_min, feet_max))

    # Building network
    if recurrent:
        if load_from_file:

            c = RecurrentClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(loaded_network_fname, epoch0 - 1),
                log=log)
        else:
            c = RecurrentClassifier(n_in, n_out, mlp_n_in, mlp_n_hidden,
                                    mlp_activation_names, lbn_n_hidden,
                                    lbn_n_out, det_activations,
                                    stoch_activations,
                                    likelihood_precision, rnn_hidden,
                                    rnn_activations, rnn_type,
                                    log=log, noise_type=noise_type)

    else:
        if network_type == network_types[0]:
            if load_from_file:

                c = Classifier.init_from_file(
                    '{0}_epoch_{1}.json'.format(
                        loaded_network_fname, epoch0 - 1),
                    log=log)
            else:

                c = Classifier(n_in, n_out, lbn_n_hidden,
                               det_activations,
                               stoch_activations, log=log,
                               likelihood_precision=likelihood_precision,
                               batch_normalization=batch_normalization,
                               mlp_n_in=mlp_n_in, mlp_n_hidden=mlp_n_hidden,
                               mlp_activation_names=mlp_activation_names,
                               bone_networks=bone_networks)
        else:

            if load_from_file:
                c = ResidualClassifier.init_from_file(
                    '{0}_epoch_{1}.json'.format(
                        loaded_network_fname, epoch0 - 1),
                    log=log)
            else:
                c = ResidualClassifier(n_in, n_out, lbn_n_hidden,
                                       det_activations,
                                       stoch_activations, log=log,
                                       likelihood_precision=likelihood_precision,
                                       batch_normalization=batch_normalization)

    print "model loaded"

    log.info("Training with data x from: {0}".format(
        data_x_from_file_name if data_x_from_file_name is not None else "files n: {0} n_impulse_2000: {1}".format(n, n_impulse_2000)))
    log.info("Training with data y from: {0}".format(
        data_y_from_file_name if data_y_from_file_name is not None else "files n: {0} n_impulse_2000: {1}".format(n, n_impulse_2000)))

    if not load_idx:
        np.savetxt('{0}/idx_train.txt'.format(opath),
                   np.asarray(idx[:train_bucket], dtype=int), fmt='%i')
        np.savetxt('{0}/idx_test.txt'.format(opath),
                   np.asarray(idx[train_bucket:], dtype=int), fmt='%i')

    # Training
    if load_from_file:
        log.info("Network loaded from file: {0}".format(loaded_network_fname))

    if recurrent:
        log.info("Network properties: n_in: {0}, mlp_n_hidden: [{1}], mlp_activation_names: "
                 "[{2}], lbn_n_hidden: [{3}], det_activations: [{4}], stoch_activations: [{5}] "
                 "rnn_hidden: {6}, rnn_activations: {7}, rnn_type: {8} "
                 "noise_type: {9}, batch_normalization: {10}".format(
                     n_in,
                     ','.join(str(e) for e in mlp_n_hidden),
                     ','.join(str(e) for e in mlp_activation_names),
                     ','.join(str(e) for e in lbn_n_hidden),
                     ','.join(str(e) for e in det_activations),
                     ','.join(str(e) for e in stoch_activations),
                     rnn_hidden, rnn_activations, rnn_type,
                     noise_type, batch_normalization))

    else:
        if network_type == network_types[0]:
            log.info("Network properties: n_in: {0}, mlp_n_hidden: [{1}], mlp_activation_names: "
                     "[{2}], lbn_n_hidden: [{3}], det_activations: [{4}], stoch_activations: [{5}] "
                     "noise_type: {6}, batch_normalization: {6}".format(
                         n_in,
                         ','.join(str(e) for e in mlp_n_hidden),
                         ','.join(str(e) for e in mlp_activation_names),
                         ','.join(str(e) for e in lbn_n_hidden),
                         ','.join(str(e) for e in det_activations),
                         ','.join(str(e) for e in stoch_activations),
                         noise_type, batch_normalization))

        elif network_type == network_types[1]:
            log.info("Network properties: n_in: {0}, "
                     "lbn_n_hidden: [{1}], det_activations: [{2}], stoch_activations: [{3}] "
                     "noise_type: {4}, batch_normalization: {5}".format(
                         n_in,
                         ','.join(str(e) for e in lbn_n_hidden),
                         ','.join(str(e) for e in det_activations),
                         ','.join(str(e) for e in stoch_activations),
                         noise_type, batch_normalization))

    # Training
    c.fit(x_train, y_train, m, n_epochs, b_size, method, fname=fname,
          epoch0=epoch0, chunk_size=chunk_size,
          save_every=save_every, sample_axis=1 if recurrent else 0)


if __name__ == '__main__':
    main()
