from LBNRNN import LBNRNN
from lbn import LBN
import cPickle, gzip
import numpy as np


if __name__ == '__main__':
    
    np.random.seed(0)
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    x_train = train_set[0]
    y_train = train_set[1]
    x_val = valid_set[0]
    y_val = valid_set[1]
    f.close()

    x_train = x_train[:500]
    y_train = x_train[:, 14*28:].copy()
    x_train = x_train[:, :14*28]
    batch_size = 100

    n_in = x_train.shape[1]
    n_hidden = [200]
    n_out = y_train.shape[1]
    det_activations = ['linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    stoch_n_hidden = [-1]
    m = 5
    lbn = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)
    rnn = VanillaRNN(self.lbn_output, self.lbn.n_out,   n_hidden, self.n_out, rnn_activation,
                        #                                                        rng=self.lbn.rng)


