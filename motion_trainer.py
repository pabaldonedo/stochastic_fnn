import numpy as np
from LBNRNN import LBNRNN_module
from rnn import VanillaRNN
from lbn import LBN


def main():

    control_path = "data/controls_1_len_61.txt"
    state_path = "data/states_1_len_61.txt"
    x = np.genfromtxt(state_path, delimiter=',')
    mux = np.mean(x,axis=0)
    stdx = np.std(x,axis=0)
    stdx[stdx==0] = 1.
    x = (x-mux)*1./stdx
    idx = np.random.permutation(x.shape[0])
    x = x[idx,:]
    #x = x.reshape(61,-1, x.shape[1])

    y = np.genfromtxt(control_path, delimiter=',')

    muy = np.mean(y,axis=0)
    stdy = np.std(y,axis=0)
    stdy[stdy==0] = 1.
    y = (y-muy)*1./stdy
    #y = y.reshape(61,-1, y.shape[1])
    y = y[idx]

    n_in = x.shape[1]
    n_hidden = [100, 50]
    n_out = y.shape[1]
    m = 5
    stoch_n_hidden = [-1]
    det_activations = ['linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    epochs = 60
    batch_size = 100
    lr = 10

    #lbn = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations,
    #                                                                stoch_n_hidden=stoch_n_hidden)
    epoch0 = 416
    fname = "network_output/networks/lbn_n_hidden_[{0}]_stoch_n_hidden_[{1}]_det_activations_[{2}]"\
            "_stoch_activations_[{3}]_lr_{4}_bsize_{5}".format(','.join(str(e) for e in n_hidden),
                                                        ','.join(str(e) for e in stoch_n_hidden),
                                                        ','.join(det_activations),
                                                        ','.join(stoch_activations), lr, batch_size)

    lbn = LBN.init_from_file("{0}_epoch_415.json".format(fname), epoch0=epoch0)

    fname = "network_output/lbn_n_hidden_[{0}]_stoch_n_hidden_[{1}]_det_activations_[{2}]"\
            "_stoch_activations_[{3}]_lr_{4}_bsize_{5}".format(','.join(str(e) for e in n_hidden),
                                                        ','.join(str(e) for e in stoch_n_hidden),
                                                        ','.join(det_activations),
                                                        ','.join(stoch_activations), lr, batch_size)
    lbn.fit(x, y, m, lr, epochs, batch_size, fname="{0}".format(fname), save_every=5)
    #n_in = x.shape[2]
    #lbn_n_hidden = [2]
    #n_out = y.shape[2]
    #det_activations = ['linear', 'linear']
    #stoch_activations = ['sigmoid', 'sigmoid']
    #stoch_n_hidden = [-1]
    #lbn_definition = {'n_in':n_in, 'n_hidden':lbn_n_hidden, 'n_out':n_out,
    #                        'det_activations':det_activations,
    #                        'stoch_activations':stoch_activations,
    #                        'stoch_n_hidden': stoch_n_hidden}

    #rnn_definition = {'n_hidden': [20], 'n_out': n_out, "activations": ['linear', 'linear']}
    
    #lbnrnn = LBNRNN_module(lbn_definition, rnn_definition)
    #lbnrnn = LBNRNN_module.init_from_file('net.json')
    

if __name__ == '__main__':
    main()