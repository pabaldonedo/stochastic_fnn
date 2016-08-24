import numpy as np
import theano
import os
import warnings
import pandas as pd
import cPickle
import sys
sys.path.append("../")
from util import log_init
from classifiers import MLPClassifier
from classifiers import RecurrentMLP
from classifiers import ResidualMLPClassifier
from classifiers import BoneResidualMLPClassifier
from classifiers import Classifier
from classifiers import Correlated2DMLPClassifier


def main():

    network_type = 'lbn'
    extra_tag = ''  # '_no_fallen_critical_learning_0_7'
    load_idx = True
    # idx_train_no_fallen.txt'
    idx_train_file = 'network_output/idx_train.txt'
    idx_test_file = 'network_output/idx_test.txt'
    assert not (load_idx and idx_train_file is None)

    network_types = ['mlp', 'residual_mlp', 'bone_residual',
                     'lbn', 'residual_lbn', 'correlated2dmlp']

    assert network_type in network_types

    use_pca = False
    pca_file = 'pca_sklearn.pkl'

    correlated_outputs = None

    if use_pca:
        assert pca_file is not None

    load_means_from_file = True

    states_file = 'data/merged_starting_states.txt'
    controls_file = 'data/merged_controls.txt'

    lagged = False

    x_info_file = 'mux_stdx.csv'
    y_info_file = 'muy_stdy.csv'

    # mean and std files:
    x_info = np.asarray(np.genfromtxt(
        x_info_file, delimiter=','), dtype=theano.config.floatX)
    y_info = np.asarray(np.genfromtxt(
        y_info_file, delimiter=','), dtype=theano.config.floatX)
    print "means loaded"
    assert not (load_means_from_file and x_info is None and y_info is None)

    y = np.asarray(pd.read_csv(
        controls_file, delimiter=',',
        header=None).values, dtype=theano.config.floatX)

    if load_means_from_file:
        muy = y_info[0]
        stdy = y_info[1]
    else:
        muy = np.mean(y, axis=0)
        stdy = np.std(y, axis=0)
    stdy[stdy == 0] = 1.

    train_size = 0.8

    print "controls loaded"
    y = (y - muy) * 1. / stdy

    if not load_idx:
        idx = np.random.permutation(y.shape[0])
        y = y[idx]
        train_bucket = int(np.ceil(y.shape[0] * train_size))
        y_train = y[:train_bucket]
        y_test = y[train_bucket:]
    else:
        idx_train = np.genfromtxt(idx_train_file, delimiter=',', dtype=int)
        idx_test = np.genfromtxt(idx_test_file, delimiter=',', dtype=int)
        y_train = y[idx_train]
        y_test = y[idx_test]

    if lagged:
        raise NotImplementedError
    else:
        x = np.asarray(pd.read_csv(
            states_file, delimiter=',',
            header=None).values, dtype=theano.config.floatX)

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

    print "states loaded"
    x = (x - mux) * 1. / stdx

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

    hidden_activation_names = ['relu', 'linear']
    # , [50, 50], [30, 30]]  # , [80, 80], [50, 50]]  # , 50]
    n_hidden = [40]

    likelihood_precision = 1

    output_activation_name = 'linear'
    # Fit options
    b_size = 100
    epoch0 = 1
    n_epochs = 1000
    lr = .001
    save_every = 10  # Log saving
    chunk_size = None  # Memory chunks
    batch_normalization = False  # TODO
    dropout = False
    l2_coeff = 0
    l1_coeff = 0
    bone_networks = False
    bone_activation_names = ['tanh']
    bone_n_hidden = [5]
    bone_type = '2d'

    # FOR LBN!!
    if network_type in [network_types[3], network_types[4]]:

        mlp_n_in = 6
        det_activations = hidden_activation_names
        stoch_activations = ['sigmoid', 'sigmoid']
        m = 20

    # Optimizer
    opt_type = 'SGD'
    method = {'type': opt_type, 'lr_decay_schedule': 'constant',
              'lr_decay_parameters': [lr],
              'momentum_type': 'nesterov', 'momentum': 0.01, 'b1': 0.9,
              'b2': 0.999, 'epsilon': 1e-8, 'rho': 0.95, 'e': 1e-8,
              'learning_rate': lr, 'dropout': dropout}

    # Load from file?
    load_from_file = False
    session_name = None
    load_different_file = False

    assert not (load_different_file and not load_from_file), "You have set load different_file to True but you are not loading any network!"

    network_name = "{0}{1}{2}_n_hidden_[{3}]_activation_[{4}]"\
        "_bsize_{5}_method_{6}_bn_{7}_dropout_{8}{9}{10}{11}{12}".\
        format('{0}_classifier'.format(network_type),
               '_bones_n_hidden[{0}]'.format(
                   ','.join(str(e) for e in bone_n_hidden)) if bone_networks else '',
               '[{0}]'.format(
            str(e) for e in bone_activation_names) if bone_networks else '',
            ','.join(str(e) for e in n_hidden),
            ','.join(str(e)
                     for e in hidden_activation_names),
            b_size,  method['type'], batch_normalization,
            dropout,
            '_lagged' if lagged else '', '_pca' if use_pca else '', '_{0}correlated'.format(correlated_outputs) if correlated_outputs is not None else '', extra_tag)

    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)
    print "Paths created"
    fname = '{0}/{1}_n_hidden_[{2}]'.format(
        opath, '{0}_classifier'.format(network_type), ','.join(str(e) for e in n_hidden))
    loaded_network_fname = '{0}/networks/{1}_n_hidden_[{2}]'.format(
        opath, '{0}_classifier'.format(network_type), ','.join(str(e) for e in n_hidden))

    if load_different_file:
        warnings.warn(
            "CAUTION: loading log and network from different path than the saving path")
#        loaded_network_folder = "residual_mlp_classifier_n_{0}_n_impulse_2000_{1}_mlp_n_hidden_[{2}]_mlp_activation_[{3}]_bsize_{4}_method_SGD_bn_False_dropout_False".format(n, n_impulse_2000,
 #                                                                                                                                                                             ','.join(
  #                                                                                                                                                                                str(e) for e in mlp_n_hidden),
   #                                                                                                                                                                           ','.join(
    #                                                                                                                                                                              str(e) for e in mlp_activation_names),
     # b_size,  method['type'])
        loaded_network_folder = "{0}_mlp_n_hidden_[{1}]_mlp_activation_[{2}]"\
            "_bsize_{3}_method_SGD_bn_{5}_dropout_{6}{7}{8}_no_fallen".\
            format(
                '{0}_classifier'.format(network_type),
                ','.join(str(e) for e in mlp_n_hidden),
                ','.join(str(e) for e in mlp_activation_names),
                b_size,  method['type'], batch_normalization,
                dropout,
                '_lagged' if lagged else '', 'pca' if use_pca else '', extra_tag)

        loaded_opath = "network_output/{0}".format(loaded_network_folder)
        assert os.path.exists(
            loaded_opath), "Trying to load a network from a non existing path; {0}".format(
            loaded_opath)

        loaded_network_name = "{0}_n_hidden_[{1}]".format('{0}_classifier'.format(network_type),
                                                          ','.join(str(e) for e in mlp_n_hidden))

        loaded_network_fname = "{0}/networks/{1}".format(
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

    if load_from_file:
        if network_type is network_types[0]:
            c = MLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    loaded_network_fname,  epoch0 - 1),
                log=log)
        elif network_type is network_types[1]:
            c = ResidualMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    loaded_network_fname,  epoch0 - 1),
                log=log)
        elif network_type is network_types[2]:
            c = BoneResidualMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    loaded_network_fname,  epoch0 - 1),
                log=log)

        elif network_type is network_type[5]:
            c = Correlated2DMLPClassifier.init_from_file(
                '{0}_epoch_{1}.json'.format(
                    loaded_network_fname, epoch0 - 1),
                log=log)
        else:
            raise NotImplementedError
    else:
        if network_type is network_types[0]:
            c = MLPClassifier(n_in, n_out, n_hidden,
                              hidden_activation_names, log=log,
                              likelihood_precision=likelihood_precision,
                              batch_normalization=batch_normalization,
                              dropout=dropout, correlated_outputs=correlated_outputs)
        elif network_type is network_types[1]:
            c = ResidualMLPClassifier(n_in, n_out, n_hidden,
                                      hidden_activation_names, log=log,
                                      likelihood_precision=likelihood_precision,
                                      batch_normalization=batch_normalization,
                                      dropout=dropout, correlated_outputs=correlated_outputs)
        elif network_type is network_types[2]:
            c = BoneResidualMLPClassifier(n_in, n_out, n_hidden,
                                          hidden_activation_names,
                                          bone_n_hidden, bone_activation_names,
                                          log=log,
                                          likelihood_precision=likelihood_precision,
                                          batch_normalization=batch_normalization,
                                          dropout=dropout, correlated_outputs=correlated_outputs)
        elif network_type is network_types[3]:
            c = Classifier(n_in, n_out, n_hidden,
                           det_activations,
                           stoch_activations, log=log,
                           likelihood_precision=likelihood_precision,
                           batch_normalization=batch_normalization,
                           mlp_n_in=mlp_n_in, mlp_n_hidden=bone_n_hidden,
                           mlp_activation_names=bone_activation_names,
                           bone_networks=bone_networks, bone_type=bone_type)

        elif network_type is network_types[5]:
            c = Correlated2DMLPClassifier(n_in, n_out, n_hidden, hidden_activation_names,
                                          likelihood_precision=likelihood_precision, log=log,
                                          batch_normalization=batch_normalization, dropout=dropout,
                                          correlated_outputs=correlated_outputs,
                                          output_activation_name=output_activation_name)
        else:
            raise NotImplementedError

    print "model loaded"

    if not load_idx:
        np.savetxt('{0}/idx_train.txt'.format(opath),
                   np.asarray(idx[:train_bucket], dtype=int), fmt='%i')
        np.savetxt('{0}/idx_test.txt'.format(opath),
                   np.asarray(idx[train_bucket:], dtype=int), fmt='%i')

    # Training
    if load_from_file:
        log.info("Network loaded from file: {0}".format(loaded_network_fname))

    if network_type in [network_types[:3], network_types[5]]:
        log.info("Network properites: n_in: {0}, n_out: {1}, n_hidden: [{2}] "
                 "hidden_activation_names: {3}, batch_normalization: {4}, "
                 "dropout: {5}, {6}".format(
                     n_in, n_out,
                     ','.join(str(e) for e in n_hidden),
                     ','.join(str(e) for e in hidden_activation_names),
                     batch_normalization, dropout,
                     'bone_n_hidden: [{0}], bone_activation_names: [{1}]'.format(','.join(str(e) for e in bone_n_hidden), ','.join(str(e) for e in bone_activation_names)) if bone_networks else ''))
    else:
        log.info("Network properites: n_in: {0}, n_out: {1}, n_hidden: [{2}] "
                 "det_activation_names: {3}, stoch_activation_names: [{4}], batch_normalization: {5}, "
                 "dropout: {6}, {7}".format(
                     n_in, n_out,
                     ','.join(str(e) for e in n_hidden),
                     ','.join(str(e) for e in det_activations),
                     ','.join(str(e) for e in stoch_activations),
                     batch_normalization, dropout,
                     'bone_n_hidden: [{0}], bone_activation_names: [{1}]'.format(','.join(str(e) for e in bone_n_hidden), ','.join(str(e) for e in bone_activation_names)) if bone_networks else ''))

    if use_pca:
        with open(pca_file, 'r') as fid:
            pca = cPickle.load(fid)
            x = pca.transform(x)

        log.info("PCA: True from file: {0}".format(pca_file))

    else:
        log.info("PCA: False")

    if network_type == network_types[3] or network_type == network_types[4]:
        c.fit(x_train, y_train, m, n_epochs, b_size, method, save_every=save_every, fname=fname, epoch0=epoch0,
              chunk_size=chunk_size, sample_axis=0,
              l2_coeff=l2_coeff, l1_coeff=l1_coeff)
    else:
        # Training
        c.fit(x_train, y_train, n_epochs, b_size, method, fname=fname,
              x_test=None, y_test=None,
              epoch0=epoch0, chunk_size=chunk_size,
              save_every=save_every, sample_axis=0,
              batch_logger=None, l2_coeff=l2_coeff, l1_coeff=l1_coeff)


if __name__ == '__main__':
    main()
