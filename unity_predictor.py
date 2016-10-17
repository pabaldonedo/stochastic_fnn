from classifiers import RecurrentClassifier
from classifiers import Classifier
from classifiers import MLPClassifier
from classifiers import RecurrentMLP
from classifiers import RNNClassifier
from classifiers import ResidualMLPClassifier
from classifiers import Correlated2DMLPClassifier
from classifiers import BoneMLPClassifier
import theano
import numpy
import warnings
import zmq
from sklearn.decomposition import PCA
import cPickle


class Predictor(object):
    """Base class for all predictors used for control in Unity"""

    def __init__(self, twod=False):
        self.twod = twod

    def set_up_means(self, mux, stdx, muy, stdy):
        """ Set up mean and standard deviation correction

        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """

        if mux.dtype is not theano.config.floatX:
            mux = numpy.asarray(mux, dtype=theano.config.floatX)
            warnings.warn("Mux dtype casted to: {0}".format(
                theano.config.floatX))
        if stdx.dtype is not theano.config.floatX:
            stdx = numpy.asarray(stdx, dtype=theano.config.floatX)
            warnings.warn("stdx dtype casted to: {0}".format(
                theano.config.floatX))

        if muy.dtype is not theano.config.floatX:
            muy = numpy.asarray(muy, dtype=theano.config.floatX)
            warnings.warn("Muy dtype casted to: {0}".format(
                theano.config.floatX))
        if stdy.dtype is not theano.config.floatX:
            stdy = numpy.asarray(stdy, dtype=theano.config.floatX)
            warnings.warn("stdy dtype casted to: {0}".format(
                theano.config.floatX))

        if self.twod:
            self.mux = mux
            self.stdx = stdx
        else:
            cols = [1] + list(range(3, 197))

            self.mux = mux[cols]
            self.stdx = stdx[cols]
            
        self.muy = muy
        self.stdy = stdy

        
    def set_up_lagged(self, lagged):
        self.lagged = lagged
        if self.lagged:
            if self.twod:
                self.x_t_1 = numpy.zeros((1,25), dtype=theano.config.floatX)
                
            else:
                self.x_t_1 = numpy.zeros((1,195), dtype=theano.config.floatX)

    def get_x(self, x_t):

        if not self.lagged:
            return x_t
        x = numpy.hstack((self.x_t_1, x_t))
        self.x_t_1 = x_t
        return x
        
class RNNPredictor(Predictor):
    """ Predictor using LBN + RNN (defined as RecurrentClassifier in classifiers.py)"""

    def __init__(self, fname, mux, stdx, muy, stdy, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(RNNPredictor, self).__init__(twod=twod)
        
        # Loads classifier from fname
        self.classifier = RecurrentClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

        
    def predict(self, x, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 1, 197)

        :return (1,1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[2]))
            x = x[:, :, cols]

        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, 1, -1)

        if pca is not None:
            x_norm = pca.transform(x_norm)
        return self.classifier.predict_one(x_norm, 1)[0] * self.stdy + self.muy


class FNNPredictor(Predictor):
    """Predictor using LBN (defined sa Classifier in classifiers.py). """

    def __init__(self, fname, mux, stdx, muy, stdy, gmm_prediction=False, lagged=False, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(FNNPredictor, self).__init__(twod=twod)
        self.classifier = Classifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.gmm_prediction = gmm_prediction
        self.set_up_lagged(lagged)

    def predict(self, x, n_out=34, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,n_out) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))

            x = x[:, cols]
        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1, -1)

        if pca is not None:
            x_t_norm = pca.transform(x_t_norm)
        x_norm = self.get_x(x_t_norm)
        if self.twod:
            return self.classifier.predict(x_norm, 1)[0] * self.stdy + self.muy
            
        if n_out < 34:
            if self.gmm_prediction:
                return numpy.hstack((self.classifier.predict_gmm(x_norm, 1) * self.stdy[:n_out] + self.muy[:n_out], 100*numpy.ones((1,34-n_out)))) 
            else:
                return numpy.hstack((self.classifier.predict(x_norm, 1)[0] * self.stdy[:n_out] + self.muy[:n_out], 100*numpy.ones((1,34-n_out)))) 
        else:
            if self.gmm_prediction:
                return self.classifier.predict_gmm(x_norm, 1) * self.stdy + self.muy
            else:
                return self.classifier.predict(x_norm, 1)[0] * self.stdy + self.muy

class MLPPredictor(Predictor):
    """ Predictor using only a classical MLP network
    (defined as MLPClassifier in classifiers.py)
    """

    def __init__(self, fname, mux, stdx, muy, stdy, lagged=False, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(MLPPredictor, self).__init__(twod=twod)
        self.classifier = MLPClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.set_up_lagged(lagged)
        
    def predict(self, x, n_out=34, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))
            x = x[:, cols]
            
        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1, -1)
        if pca is not None:
            x_t_norm = pca.transform(x_t_norm)

        x_norm = self.get_x(x_t_norm)
       
        if self.twod:
            return self.classifier.predict(x_norm) * self.stdy + self.muy
            
        if n_out < 34:
            return numpy.hstack((self.classifier.predict(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict(x_norm) * self.stdy + self.muy


class CorrelatedPredictor(Predictor):
    """ Predictor using only a classical MLP network
    (defined as MLPClassifier in classifiers.py)
    """

    def __init__(self, fname, mux, stdx, muy, stdy):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(CorrelatedPredictor, self).__init__(twod=True)
        self.classifier = Correlated2DMLPClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.lagged = False
        
    def predict(self, x, n_out=3, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,3) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1, -1)
        if pca is not None:
            x_t_norm = pca.transform(x_t_norm)
        
        x_norm = self.get_x(x_t_norm)

        return self.classifier.predict(x_norm) * self.stdy + self.muy
            
        
class RecurrentMLPPredictor(Predictor):
    """ Predictor using only a classical MLP network
    (defined as MLPClassifier in classifiers.py)
    """

    def __init__(self, fname, mux, stdx, muy, stdy, lagged=False, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(RecurrentMLPPredictor, self).__init__(twod=twod)
        
        self.classifier = RecurrentMLP.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.set_up_lagged(lagged)

    def predict(self, x, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))
            x = x[:,:, cols]
            
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, 1, -1)
        if pca is not None:
            x_norm = pca.transform(x_norm)
        return self.classifier.predict_one(x_norm)[0] * self.stdy + self.muy



class RNNPredictor(Predictor):
    """ Predictor using only a classical RNN network
    (defined as RNNClassifier in classifiers.py)
    """

    def __init__(self, fname, mux, stdx, muy, stdy, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(RNNPredictor, self).__init__(twod=twod)
        
        self.classifier = RNNClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.set_up_lagged(lagged)
        
    def predict(self, x, n_out=34, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))
            x = x[:,:, cols]
            
        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1,1, -1)

        x_norm = self.get_x(x_t_norm)
        if pca is not None:
            x_norm = self.pca.transform(x_norm)

        if self.twod:
            return self.classifier.predict_one(x_norm) * self.stdy + self.muy
            
        if n_out < 34:
            return numpy.hstack((self.classifier.predict_one(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict_one(x_norm) * self.stdy + self.muy


class ResidualPredictor(Predictor):
    """ Predictor using only a classical MLP network
    (defined as MLPClassifier in classifiers.py)
    """

    def __init__(self, fname, mux, stdx, muy, stdy, lagged=False, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.
        """
        super(ResidualPredictor, self).__init__(twod=twod)
        self.classifier = ResidualMLPClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.set_up_lagged(lagged)

    def predict(self, x, n_out=34, pca=None):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))
            x = x[:, cols]
            
        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1, -1)

        if pca is not None:
            x_t_norm = pca.transform(x_t_norm)
        
        x_norm = self.get_x(x_t_norm)

        if self.twod:
            return self.classifier.predict(x_norm) * self.stdy + self.muy
        
        if n_out < 34:
            return numpy.hstack((self.classifier.predict(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict(x_norm) * self.stdy + self.muy

class BoneMLPPredictor(Predictor):

    def __init__(self, fname, mux, stdx, muy, stdy, lagged=False, twod=False):
        super(BoneMLPPredictor, self).__init__(twod=twod)
        self.classifier = BoneMLPClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)
        self.set_up_lagged(lagged)

    def predict(self, x, n_out=34, pca=None):
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        if not self.twod:
            cols = [1] + list(range(3, x.shape[1]))
            x = x[:, cols]
            
        x_t_norm = (x - self.mux) * 1. / self.stdx
        x_t_norm.reshape(1, -1)
        if pca is not None:
            x_t_norm = pca.transform(x_t_norm)

        x_norm = self.get_x(x_t_norm)
       
        if self.twod:
            return self.classifier.predict(x_norm) * self.stdy + self.muy
            
        if n_out < 34:
            return numpy.hstack((self.classifier.predict(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict(x_norm) * self.stdy + self.muy

        
    
class UnityMessenger(object):
    """
    Opens a ZMQ socket to communicate with c++ Unity program
    """

    def __init__(self, fname, mux, stdx, muy, stdy, classifier_type, port=5555, n_out=34, gmm_prediction=False, lagged=False, pca=False, pca_file=None, twod=False):
        """
        :type fname: string.
        :param fname: Filename (with path) containing the classifier definition.

        :type: mux: numpy.array.
        :param mux: mean to be subtracted from the input array.

        :type stdx: numpy.array.
        :param stdx: standard deviation to scale the input array.

        :type muy: numpy.array.
        :param muy: mean to be added to the output array.

        :type stdy: numpy.array.
        :param stdy: standar deveiation to scale the output array.

        :type classifier_type: string.
        :param classifier_type: defines which classifier to be used. One of
                                "Recurrent, "Classifier" or "MLP".

        :type port: int.
        :param port: port number used in zeromq.
        """
        # Sets up socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))
        classifier_types = ['Recurrent', 'Classifier', 'MLP', 'RecurrentMLP', 'RNN', 'ResidualMLP', 'Correlated2DMLP', 'BoneMLP']

        self.n_out = n_out
        self.gmm_prediction = gmm_prediction
        self.use_pca = pca
        assert not (pca_file is None and self.use_pca)
        if self.use_pca:
            with open(pca_file, 'r') as fid:
                self.pca = cPickle.load(fid)
        else:
            self.pca = None
        # Sets up predictor
        if classifier_type == classifier_types[0]:
            assert not self.gmm_prediction, "GMM prediction not available in this mode"
            self.predictor = RNNPredictor(fname, mux, stdx, muy, stdy, twod=twod)
            self.recurrent = True
        elif classifier_type == classifier_types[1]:
            self.predictor = FNNPredictor(fname, mux, stdx, muy, stdy, gmm_prediction=self.gmm_prediction, lagged=lagged, twod=twod)
            self.recurrent = False
        elif classifier_type == classifier_types[2]:
            assert not self.gmm_prediction, "GMM prediction not available in this mode"
            self.predictor = MLPPredictor(fname, mux, stdx, muy, stdy, lagged=lagged, twod=twod)
            self.recurrent = False
        elif classifier_type == classifier_types[3]:
            assert not self.gmm_prediction, "GMM prediction not available in this mode"
            self.predictor = RecurrentMLPPredictor(fname, mux, stdx, muy, stdy, twod=twod)
            self.recurrent = True
        elif classifier_type == classifier_types[4]:
            assert not self.gmm_prediction, "GMM prediction not available in this mode"
            self.predictor = RNNPredictor(fname, mux, stdx, muy, stdy, twod=twod)
            self.recurrent = True
        elif classifier_type == classifier_types[5]:
            assert not self.gmm_prediction, "GMM prediction not avaiable in this mode"
            self.recurrent = False
            self.predictor = ResidualPredictor(fname, mux, stdx, muy, stdy, lagged=lagged, twod=twod)
        elif classifier_type == classifier_types[6]:
            assert not self.gmm_prediction, "GMM prediction not avaiable in this mode"
            self.recurrent = False
            self.predictor = CorrelatedPredictor(fname, mux, stdx, muy, stdy)
        elif classifier_type == classifier_types[7]:
            self.recurrent = False
            self.predictor = BoneMLPPredictor(fname, mux, stdx, muy, stdy, twod=twod)
        else:
            raise NotImplementedError

    def listen(self):
        print "Listening starts"
        while True:
            #  Wait for next request from client
            message = self.socket.recv()
           # print "Received request: %s" % message

            x = numpy.fromstring(message, sep=',')
            # Do the work
            if self.recurrent:
                x = x.reshape(1, 1, -1)
            else:
                x = x.reshape(1, -1)

            y = self.predictor.predict(x,n_out=self.n_out, pca=self.pca).flatten()
            print y

            #  Send reply back to client
            self.socket.send(bytes(str(y)[1:-1]))

if __name__ == '__main__':
    port = 5555
    fname = "network_output/mlp_n_13_n_hidden_[150]_epoch_300.json"
    n_out = 34
    classifier_type = 'MLP'
    twod = False

    #x_info = numpy.genfromtxt('mux_stdx.csv', delimiter=',')
   # y_info = numpy.genfromtxt('muy_stdy.csv', delimiter=',')
    
#    x_info = numpy.genfromtxt('mux_stdx_lagged_n_16.csv', delimiter=',')
    x_info = numpy.genfromtxt('mux_stdx_n_16_n_impulse_2000_5.csv', delimiter=',')
    y_info = numpy.genfromtxt('muy_stdy_n_16_n_impulse_2000_5.csv', delimiter=',')
    #    x_info = numpy.genfromtxt('sample_clipped_mux_stdx_n_16_n_impules_2000_5.csv',delimiter=',')
#    y_info = numpy.genfromtxt('sample_clipped_muy_stdy_n_16_n_impules_2000_5.csv', delimiter=',')
    #x_info = numpy.genfromtxt('mux_stdx_n_16_n_impulse_2000_5.csv', delimiter=',')
    #y_info = numpy.genfromtxt('muy_stdy_n_16_n_impulse_2000_5.csv', delimiter=',')
#    x_info = numpy.genfromtxt('mux_stdx_n_13_n_impulse_2000_5.csv', delimiter=',')
#    y_info = numpy.genfromtxt('muy_stdy_n_13_n_impulse_2000_5.csv', delimiter=',')
    #x_info = numpy.genfromtxt('mux_stdx_n_13.csv', delimiter=',')
    #y_info = numpy.genfromtxt('muy_stdy_n_13.csv', delimiter=',')
    lagged = False
    gmm_prediction = False
    pca = False
    pca_file = ""#"2d/pca_sklearn.pkl"
    mux = x_info[0]
    stdx = x_info[1]
    stdx[stdx==0] = 1

    muy = y_info[0]
    stdy = y_info[1]
    messenger = UnityMessenger(
        fname, mux, stdx, muy, stdy, classifier_type, port=port, n_out=n_out, gmm_prediction=gmm_prediction, lagged=lagged, pca=pca, pca_file=pca_file, twod=twod)
    messenger.listen()
