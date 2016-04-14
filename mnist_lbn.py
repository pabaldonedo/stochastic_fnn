import cPickle, gzip
import numpy as np
from lbn import LBN

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
    #n = LBN(n_in, n_hidden, n_out, det_activations, stoch_activations, stoch_n_hidden)
    n = LBN.init_from_file("last_network.json")
    #y_hat = n.predict(x_train,m)
    # for i in xrange(10):
    #     plt.figure()
    #     compound = np.zeros((28,28*m))
    #     for k in xrange(m):
    #         compound[:14,k*28:(k+1)*28] = x_train[i].reshape(14,28)
    #         compound[14:,k*28:(k+1)*28] = y_hat[k][i].reshape(14,28)
    #     plt.imshow(compound, cmap='gray')
    #     plt.show()

    # y_hat = n.predict(x_val[:10,:14*28],m)

    # for i in xrange(10):
    #     plt.figure()
    #     compound = np.zeros((28,28*(m+1)))
    #     for k in xrange(m+1):
    #         if k == 0:
    #             compound[:,k*28:(k+1)*28] = x_val[i].reshape(28,28)
    #         else:
    #             compound[:14,k*28:(k+1)*28] = x_val[i][:14*28].reshape(14,28)
    #             compound[14:,k*28:(k+1)*28] = y_hat[k-1][i].reshape(14,28)
    #     plt.imshow(compound, cmap='gray')
    #     plt.show()
    epochs = 100
    lr = 0.1
    n.fit(x_train,y_train,m, lr,epochs, batch_size)
    y_hat = n.predict(x_train[0].reshape(1,-1), m)
    y_points = np.linspace(-1,10).reshape(1,-1,1)

    #distribution = np.sum(np.exp(-0.5*np.sum((y_points-y_hat)**2, axis=2)), axis=0)*1./(m*np.sqrt((2*np.pi)**y_hat.shape[2]))
    #plt.plot(y_points[0,:,0], distribution)
    #plt.show()
