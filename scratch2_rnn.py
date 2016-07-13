import numpy as np
import theano
import os
import warnings
from util import load_states
from util import load_controls
from util import log_init
from util import load_files
from classifiers import RNNClassifier


def main():

    load_means_from_file = True
    # mean and std files:
    x_info = np.asarray(np.genfromtxt(
        'mux_stdx_n_16_n_impulse_2000_5.csv', delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(
        'muy_stdy_n_16_n_impulse_2000_5.csv', delimiter=','), dtype=theano.config.floatX)

    assert not (load_means_from_file and x_info is None and y_info is None)
    # Number of datasets
    n = 16
    n_impulse_2000 = 0

    # Only for NO recurrent
    feet_learning = False
    feet_min = 50
    feet_max = 100

    assert not (
        feet_learning and recurrent), "Feet learning and recurrent cannot be true at the same time"

    # Load data
    seq_len = 61

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

    y = (y - muy) * 1. / stdy
    y = y.reshape(seq_len, -1, y.shape[1])
    idx = np.random.permutation(y.shape[1])

    y = y[:, idx, :-4]
    train_bucket = int(np.ceil(y.shape[1] * train_size))
    y_train = y[:, :train_bucket]
    y_test = y[:, train_bucket:]

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

    x = (x - mux) * 1. / stdx
    x = x.reshape(seq_len, -1, x.shape[1])

    cols = [1] + list(range(3, x.shape[2]))
    x = x[:, :, cols]
    x = x[:, idx, :]

    x_train = x[:, :train_bucket]
    x_test = x[:, train_bucket:]
    n_in = x.shape[2]
    n_out = y.shape[2]

    # RNN definiton + LBN n_out if RNN is the final layer
    rnn_type = "LSTM"
    rnn_hidden = [150]
    rnn_activations = [['sigmoid', 'tanh', 'sigmoid',
                        'sigmoid', 'tanh'], 'linear']  # ['sigmoid', 'linear']

    likelihood_precision = 0.1

    # Fit options
    b_size = 100
    epoch0 = 1
    n_epochs = 50000
    lr = 10.
    save_every = 10  # Log saving
    chunk_size = None  # Memory chunks

    # Optimizer
    opt_type = 'SGD'
    method = {'type': opt_type, 'lr_decay_schedule': 'constant',
              'lr_decay_parameters': [lr],
              'momentum_type': 'nesterov', 'momentum': 0.1, 'b1': 0.9,
              'b2': 0.999, 'epsilon': 1e-8, 'rho': 0.95, 'e': 1e-8,
              'learning_rate': lr}

    # Load from file?
    load_from_file = False
    session_name = None
    load_different_file = False

    assert not (load_different_file and not load_from_file), "You have set load different_file to True but you are not loading any network!"

    # Saving options

    network_name = "{0}_n_{1}_n_impulse_2000_{2}_"\
        "rnn_hidden_[{3}]_rnn_activations_[{4}]_bsize_{5}_method_{6}_bn_False".\
        format(
                   rnn_type,
                   n, n_impulse_2000,
                   ','.join(str(e) for e in rnn_hidden),
                   ','.join(str(e) for e in rnn_activations),
                   b_size, method['type'])

    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)
    fname = '{0}/{1}_rnn_hidden_[{2}]'.format(
        opath, rnn_type, ','.join(str(e) for e in rnn_hidden))
    loaded_network_fname = '{0}/networks/{1}_rnn_hidden_[{2}]'.format(
        opath, rnn_type, ','.join(str(e) for e in rnn_hidden))

    if load_different_file:
        warnings.warn(
            "CAUTION: loading log and network from different path than the saving path")
        loaded_network_folder = ""
        loaded_opath = "network_output/{0}".fromat(loaded_network_folder)
        assert os.path.exists(
            loaded_opath), "Trying to load a network from a non existing path; {0}".format(loaded_opath)

        loaded_network_name = "mlp_n_hidden_[150]"
        loaded_network_fname = "{0}/networks/{1}".format(
            loaded_opath, loaded_network_name)

    else:
        loaded_opath = opath

    # LOGGING
    log, session_name = log_init(
        opath, session_name=session_name if load_from_file else None)

    if feet_learning:
        log.warning("Using feet learning.\nFeet min: {0}\nFeet max: {1}".format(
            feet_min, feet_max))

    if load_from_file:
        c = RNNClassifier.init_from_file(
            '{0}_epoch_{1}.json'.format(loaded_network_fname, epoch0 - 1),
            log=log)
    else:
        c = RNNClassifier(n_in, n_out, likelihood_precision,
                          rnn_hidden, rnn_activations, rnn_type)

    # Training
    c.fit(x_train, y_train, n_epochs, b_size, method, fname=fname,
          x_test=x_test, y_test=y_test,
          epoch0=epoch0, chunk_size=chunk_size,
          save_every=save_every, sample_axis=1)


if __name__ == '__main__':
    main()
