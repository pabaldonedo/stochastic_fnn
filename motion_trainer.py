import numpy as np
from LBNRNN import LBNRNN_module
from rnn import VanillaRNN
from lbn import LBN
import logging
import petname
import os
from util import load_states
from util import load_controls


def log_init(path):
    session_name = petname.Name()
    if not os.path.exists("{0}/logs".format(path)):
        os.makedirs("{0}/logs".format(path))

    while os.path.isfile('{0}/logs/{1}.log'.format(path,session_name)):
        session_name = petname.Name()

    logging.basicConfig(level=logging.INFO, filename="{0}/logs/{1}.log".format(path, session_name),
                        format="%(asctime)s %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S")
    log = logging.getLogger(session_name)
    return log, session_name

def main():

    n = 4
    x = load_states(n)
    mux = np.mean(x,axis=0)
    stdx = np.std(x,axis=0)
    stdx[stdx==0] = 1.
    x = (x-mux)*1./stdx
    idx = np.random.permutation(x.shape[0])
    x = x[idx,:]
    #x = x.reshape(61,-1, x.shape[1])

    y = load_controls(n)

    muy = np.mean(y,axis=0)
    stdy = np.std(y,axis=0)
    stdy[stdy==0] = 1.
    y = (y-muy)*1./stdy
    #y = y.reshape(61,-1, y.shape[1])
    y = y[idx]

    n_in = x.shape[1]
    n_hidden = [100, 50]
    n_out = y.shape[1]
    m = 10
    stoch_n_hidden = [-1]
    det_activations = ['linear', 'linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    epochs = 10000
    batch_size = 100
    lr = 10
    save_every = 10


    opath = "network_output/lbn_n_hidden_[{0}]_stoch_n_hidden_[{1}]_det_activations_[{2}]"\
            "_stoch_activations_[{3}]_bsize_{4}_m_{5}".format(','.join(str(e) for e in n_hidden),
                                                        ','.join(str(e) for e in stoch_n_hidden),
                                                        ','.join(det_activations),
                                                        ','.join(stoch_activations), batch_size,
                                                        m)
    
    if not os.path.exists(opath):
        os.makedirs(opath)
    log, session_name = log_init(opath)

    log.info('States and controls 1 to {0} loaded'.format(n))
    log.info('mux: {0}, stdx: {1}, muy: {2}, stdy: {3}'.format(mux.tolist(), stdx.tolist(),
                                                            muy.tolist(), stdy.tolist() ))



    #lbn = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations,
    #                                                                stoch_n_hidden=stoch_n_hidden,
#                                                                    log=log,
#                                                                    session_name=session_name)
    epoch0 = 81
    lbn_file = "{0}/networks/lbn_n_hidden_[{1}]_stoch_n_hidden_[{2}]_det_activations_[{3}]"\
            "_stoch_activations_[{4}]_bsize_{5}_m_{6}".format(opath,
                                                        ','.join(str(e) for e in n_hidden),
                                                        ','.join(str(e) for e in stoch_n_hidden),
                                                        ','.join(det_activations),
                                                        ','.join(stoch_activations), batch_size,
                                                        m)

    lbn = LBN.init_from_file("{0}_epoch_80.json".format(lbn_file), epoch0=epoch0,
                                                                        log=log,
                                                                        session_name=session_name)

    fname = "{0}/lbn_n_hidden_[{1}]_stoch_n_hidden_[{2}]_det_activations_[{3}]"\
            "_stoch_activations_[{4}]_bsize_{5}_m_{6}".format(opath,
                                                        ','.join(str(e) for e in n_hidden),
                                                        ','.join(str(e) for e in stoch_n_hidden),
                                                        ','.join(det_activations),
                                                        ','.join(stoch_activations), batch_size,
                                                        m)
    lbn.fit(x, y, m, lr, epochs, batch_size, fname=fname, save_every=save_every)
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