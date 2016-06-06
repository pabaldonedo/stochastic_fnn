import theano.tensor as T
import theano
import numpy as np
import logging
import json
import os
from util import load_states
from util import load_controls
from util import log_init
from util import flatten
from classifiers import RecurrentClassifier
from classifiers import Classifier

def main():

    #Number of datasets
    n = 13
    #RNN on top of LBN
    recurrent = True
    seq_len = 61
    train_size = 0.8
    x = load_states(n)
    mux = np.mean(x,axis=0)
    stdx = np.std(x,axis=0)
    stdx[stdx==0] = 1.
    x = (x-mux)*1./stdx
    if recurrent:
        x = x.reshape(seq_len, -1, x.shape[1])

        idx = np.random.permutation(x.shape[1])
        cols = [1]+list(range(3,x.shape[2]))
        x = x[:,:,cols]
        x = x[:,idx,:]
        train_bucket = int(np.ceil(x.shape[1]*train_size))
        x_train = x[:,:train_bucket]
        x_test = x[:,train_bucket:]

    else:
        idx = np.random.permutation(x.shape[0])
        cols = [1]+list(range(3,x.shape[1]))
        x = x[:,cols]
        x = x[idx]
        train_bucket = int(np.ceil(x.shape[0]*train_size))
        x_train = x[:train_bucket]
        x_test = x[train_bucket:]

    y = load_controls(n)
    muy = np.mean(y,axis=0)
    stdy = np.std(y,axis=0)
    stdy[stdy==0] = 1.
    y = (y-muy)*1./stdy
    if recurrent:
        y = y.reshape(seq_len, -1, y.shape[1])
        y = y[:,idx,:-4]
        y_train = y[:,:train_bucket]
        y_test = y[:,train_bucket:]
        n_in = x.shape[2]
        n_out = y.shape[2]
    else:
        y = y[idx,:-4]
        y_train = y[:train_bucket]
        y_test = y[train_bucket:]
        n_in = x.shape[1]
        n_out = y.shape[1]

    #MLP definition
    mlp_activation_names = ['sigmoid']
    mlp_n_in = 13
    mlp_n_hidden = [10]
    
    #LBN definition
    lbn_n_hidden =  [150, 100, 50]
    det_activations = ['linear', 'linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    likelihood_precision = 0.1
    m = 10

    #RNN definiton + LBN n_out if RNN is the final layer
    rnn_type = "LSTM"
    rnn_hidden = [30]
    rnn_activations = [['sigmoid', 'tanh', 'sigmoid', 'sigmoid', 'tanh'], 'linear'] #['sigmoid', 'linear']
    lbn_n_out = 50
    noise_type = 'additive'

    #Fit options
    b_size = 100
    epoch0 = 1
    n_epochs = 100
    lr = 1
    save_every = 1 #Log saving
    chunk_size = 2000 #Memory chunks
    #Optimizer
    opt_type = 'SGD'
    method={'type':opt_type, 'lr_decay_schedule':'constant', 'lr_decay_parameters':[lr],
            'momentum_type': 'nesterov', 'momentum': 0.01, 'b1': 0.9, 'b2':0.999, 'e':1e-6,
            'learning_rate':lr}
    
    #Saving options
    network_name = "{0}_n_{1}_mlp_hidden_[{2}]_mlp_activation_[{3}]_lbn_n_hidden_[{4}]"\
                    "_det_activations_[{5}]_stoch_activations_[{6}]_m_{7}_noise_type_{8}_bsize_{9}_method_{10}".\
                                                    format(
                                                        'recurrentclassifier_{0}'.format(rnn_type) if recurrent else 
                                                                                    'classifier',
                                                    n,
                                                    ','.join(str(e) for e in mlp_n_hidden),
                                                    ','.join(str(e) for e in mlp_activation_names),
                                                    ','.join(str(e) for e in lbn_n_hidden),
                                                    ','.join(str(e) for e in det_activations),
                                                    ','.join(str(e) for e in stoch_activations),
                                                    m, noise_type, b_size, method['type'])
    network_name = "TESTING"
    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)
    fname = '{0}/{1}'.format(opath, network_name)
    network_fname = '{0}/networks/{1}'.format(opath, network_name)
    
    #LOGGING
    log, session_name = log_init(opath, session_name='lang')
        
    #Building network
    if recurrent:
        c = RecurrentClassifier(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names,
                                lbn_n_hidden,
                                lbn_n_out, det_activations, stoch_activations, likelihood_precision,
                                rnn_hidden, rnn_activations, rnn_type, log=log, noise_type=noise_type)
        #c = RecurrentClassifier.init_from_file('{0}_epoch_{1}.json'.format(network_fname, epoch0-1), log=log)

    else: 
       # c = Classifier.init_from_file('{0}_epoch_{1}.json'.format(network_fname, epoch0-1), log=log)

        c = Classifier(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                                                            det_activations,
                                                            stoch_activations, log=log,
                                                            likelihood_precision=likelihood_precision)

    #Training
    f = c.fit(x_train, y_train,m,n_epochs, b_size, method, fname=fname, epoch0=epoch0,
                                    x_test=x_test, y_test=y_test, chunk_size=chunk_size,
                                    save_every=save_every, sample_axis=1 if recurrent else 0)


if __name__ == '__main__':
    main()

