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


def main():

    network_type = 'residual'

    network_types = ['mlp', 'residual']

    assert network_type in network_types

    load_means_from_file = True
    sampled_clipped = False
    if sampled_clipped:
        print "WARNING USING CLIPPED"

    x_info_file = 'mux_stdx_n_16_n_impulse_2000_5.csv'#'sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv'
    y_info_file = 'muy_stdy_n_16_n_impulse_2000_5.csv'#'sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv'

    # mean and std files:
    x_info = np.asarray(np.genfromtxt(x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(y_info_file, delimiter=','), dtype=theano.config.floatX)
    print "means loaded"
    assert not (load_means_from_file and x_info is None and y_info is None)
    # Number of datasets
    n = 16
    n_impulse_2000 = 0

    # RNN on top of MLP
    recurrent = False

    # Only for NO recurrent
    feet_learning = False
    feet_min = 50
    feet_max = 100

    assert not (
        feet_learning and recurrent), "Feet learning and recurrent cannot be true at the same time"

    # Load data
    seq_len = 61

    if sampled_clipped:
        y = np.asarray(pd.read_csv(
            'data/sample_clipped_controls_n_16_n_impulse_2000_5.txt', delimiter=',',
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

    print "controsl loaded"
    y = (y - muy) * 1. / stdy
    if recurrent:
        y = y.reshape(seq_len, -1, y.shape[1])
        idx = np.random.permutation(y.shape[1])

        y = y[:, idx, :-4]
        train_bucket = int(np.ceil(y.shape[1] * train_size))
        y_train = y[:, :train_bucket]
        y_test = y[:, train_bucket:]

    else:
        idx = np.random.permutation(y.shape[0])
        y = y[idx, :-4]
        train_bucket = int(np.ceil(y.shape[0] * train_size))
        y_train = y[:train_bucket]
        y_test = y[train_bucket:]

    if sampled_clipped:
        x = np.asarray(pd.read_csv(
            'data/sample_clipped_states_n_16_n_impulse_2000_5.txt', delimiter=',', header=None).values, dtype=theano.config.floatX)
    else:
        x = load_states(n)
        if n_impulse_2000 > 0:
            x_impulse = load_files(n_impulse_2000, 'states_impulse_2000')
            x = np.vstack((x, x_impulse))

    if load_means_from_file:
        mux = x_info[0]
        stdx = x_info[1]
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

        cols = [1] + list(range(3, x.shape[2]))
        x = x[:, :, cols]
        x = x[:, idx, :]

        x_train = x[:, :train_bucket]
        x_test = x[:, train_bucket:]
        n_in = x.shape[2]
        n_out = y.shape[2]
    else:

        cols = [1] + list(range(3, x.shape[1]))
        x = x[:, cols]
        x = x[idx]
        x_train = x[:train_bucket]
        x_test = x[train_bucket:]
        n_in = x.shape[1]
        n_out = y.shape[1]

    print "Data ready to go"
    mlp_activation_names = ['sigmoid', 'sigmoid']#, 'sigmoid']
    mlp_n_hidden = [150, 100]#, 50]
    likelihood_precision = 0.1

    # RNN definiton + LBN n_out if RNN is the final layer
    rnn_type = "LSTM"
    rnn_hidden = [30]
    rnn_activations = [['sigmoid', 'tanh', 'sigmoid',
                        'sigmoid', 'tanh'], 'linear']  # ['sigmoid', 'linear']

    # Fit options
    b_size = 100
    epoch0 = 51
    n_epochs = 1000
    lr = .001
    save_every = 10  # Log saving
    chunk_size = None  # Memory chunks
    batch_normalization = False  # TODO
    dropout = False

    # Optimizer
    opt_type = 'SGD'
    method = {'type': opt_type, 'lr_decay_schedule': 'constant',
              'lr_decay_parameters': [lr],
              'momentum_type': 'nesterov', 'momentum': 0.01, 'b1': 0.9,
              'b2': 0.999, 'epsilon': 1e-8, 'rho': 0.95, 'e': 1e-8,
              'learning_rate': lr, 'dropout': dropout}

    # Load from file?
    load_from_file = True
    session_name = 'eloy'
    load_different_file = False

    assert not (load_different_file and not load_from_file), "You have set load different_file to True but you are not loading any network!"

    # Saving options
    if recurrent:

        network_name = "{0}_n_{1}_n_impulse_2000_{2}_mlp_n_hidden_[{3}]_mlp_activation_[{4}]"\
            "rnn_hidden_[{5}]_rnn_activations_[{6}]_bsize_{7}_method_{8}_bn_{9}_dropout_{10}".\
            format(
                       'recurrent_mlp_classifier',
                       n, n_impulse_2000,
                       ','.join(str(e) for e in mlp_n_hidden),
                       ','.join(str(e) for e in mlp_activation_names),
                       ','.join(str(e) for e in rnn_hidden),
                       ','.join(str(e) for e in rnn_activations),
                       b_size, method['type'], batch_normalization, dropout)

    else:
        network_name = "{0}_n_{1}_n_impulse_2000_{2}_mlp_n_hidden_[{3}]_mlp_activation_[{4}]"\
            "_bsize_{5}_method_{6}_bn_{7}_dropout_{8}{9}".\
            format(
                       'mlp_classifier' if network_type is network_types[0] else 'residual_mlp_classifier',
                       n, n_impulse_2000,
                       ','.join(str(e) for e in mlp_n_hidden),
                       ','.join(str(e) for e in mlp_activation_names),
                       b_size,  method['type'], batch_normalization,
                       dropout,
                       '_sampled_clipped' if sampled_clipped else '')

    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)
    print "Paths created"
    fname = '{0}/{1}_n_hidden_[{2}]'.format(
        opath, 'recurrent_mlp' if recurrent else 'mlp_classifier' if network_type is network_types[0] else 'residual_mlp_classifier', ','.join(str(e) for e in mlp_n_hidden))
    loaded_network_fname = '{0}/networks/{1}_n_hidden_[{2}]'.format(
        opath,  'recurrent_mlp' if recurrent else 'mlp_classifier' if network_type is network_types[0] else 'residual_mlp_classifier', ','.join(str(e) for e in mlp_n_hidden))

    if load_different_file:
        warnings.warn(
            "CAUTION: loading log and network from different path than the saving path")
        loaded_network_folder = "residual_mlp_classifier_n_{0}_n_impulse_2000_{1}_mlp_n_hidden_[{2}]_mlp_activation_[{3}]_bsize_{4}_method_SGD_bn_False_dropout_False".format(n, n_impulse_2000,
                                                                                                                                                       ','.join(
                                                                                                                                                           str(e) for e in mlp_n_hidden),
                                                                                                                                                       ','.join(
                                                                                                                                                           str(e) for e in mlp_activation_names),
                                                                                                                                                       b_size,  method['type'])

        loaded_opath = "network_output/{0}".format(loaded_network_folder)
        assert os.path.exists(
            loaded_opath), "Trying to load a network from a non existing path; {0}".format(loaded_opath)

        loaded_network_name = "residual_mlp_classifier_n_hidden_[150,100]"#"mlp_classifier_n_{0}_n_impulse_2000_{1}_mlp_n_hidden_[{2}]_mlp_activation_[{3}]_bsize_{4}_method_{5}".format(n, n_impulse_2000,
                              #                                                                                                              ','.join(
#                                                                                                                                                str(e) for e in mlp_n_hidden),
#                                                                                                                                            ','.join(
#                                                                                                                                                str(e) for e in mlp_activation_names),
#                                                                                                                                            b_size,  method['type'])

        loaded_network_fname = "{0}/networks/{1}".format(
            loaded_opath, loaded_network_name)

    else:
        loaded_opath = opath

    # LOGGING
    log, session_name = log_init(
        opath, session_name=session_name if load_from_file else None)
    print "LOG generated"
    if load_means_from_file:
        log.info('Loading means from x: {0}\ny: {1}'.format(x_info_file, y_info_file))
    else:
        log.info('Means and stds from data')

    if feet_learning:
        log.warning("Using feet learning.\nFeet min: {0}\nFeet max: {1}".format(
            feet_min, feet_max))

    if recurrent:
        if load_from_file:
            c = RecurrentMLP.init_from_file(
                '{0}_epoch_{1}.json'.format(loaded_network_fname, epoch0 - 1),
                log=log)
        else:
            c = RecurrentMLP(n_in, n_out, mlp_n_hidden, mlp_activation_names,
                             rnn_hidden, rnn_activations, rnn_type,
                             likelihood_precision=likelihood_precision,
                             batch_normalization=batch_normalization,
                             dropout=dropout)

    else:
        if load_from_file:
            if network_type is network_types[0]:
                c = MLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(loaded_network_fname,  epoch0 - 1),
                log=log)
            elif network_type is network_types[1]:
                c = ResidualMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(loaded_network_fname,  epoch0 - 1),
                log=log)   
        else:
            if network_type is network_types[0]:
                c = MLPClassifier(n_in, n_out, mlp_n_hidden,
                              mlp_activation_names, log=log,
                              likelihood_precision=likelihood_precision,
                              batch_normalization=batch_normalization,
                              dropout=dropout)
            elif network_type is network_types[1]:
                c = ResidualMLPClassifier(n_in, n_out, mlp_n_hidden,
                              mlp_activation_names, log=log,
                              likelihood_precision=likelihood_precision,
                              batch_normalization=batch_normalization,
                              dropout=dropout)          

    print "model loaded"
    # Training
    c.fit(x_train, y_train, n_epochs, b_size, method, fname=fname,
          x_test=x_test, y_test=y_test,
          epoch0=epoch0, chunk_size=chunk_size,
          save_every=save_every, sample_axis=1 if recurrent else 0)


if __name__ == '__main__':
    main()