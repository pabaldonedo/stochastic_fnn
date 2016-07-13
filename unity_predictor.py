from classifiers import RecurrentClassifier
from classifiers import Classifier
from classifiers import MLPClassifier
from classifiers import RecurrentMLP
from classifiers import RNNClassifier
import theano
import numpy
import warnings
import zmq


class Predictor(object):
    """Base class for all predictors used for control in Unity"""

    def __init__(self):
        pass

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

        cols = [1] + list(range(3, 197))

        self.mux = mux[cols]
        self.stdx = stdx[cols]
        self.muy = muy
        self.stdy = stdy


class RNNPredictor(Predictor):
    """ Predictor using LBN + RNN (defined as RecurrentClassifier in classifiers.py)"""

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
        # Loads classifier from fname
        self.classifier = RecurrentClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

    def predict(self, x):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 1, 197)

        :return (1,1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        cols = [1] + list(range(3, x.shape[2]))
        x = x[:, :, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, 1, -1)
        return self.classifier.predict_one(x_norm, 1)[0] * self.stdy + self.muy


class FNNPredictor(Predictor):
    """Predictor using LBN (defined sa Classifier in classifiers.py). """

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
        self.classifier = Classifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

    def predict(self, x, n_out=34):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,n_out) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        cols = [1] + list(range(3, x.shape[1]))

        x = x[:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, -1)

        if n_out < 34:
            return numpy.hstack((self.classifier.predict(x_norm, 1)[0] * self.stdy[:n_out] + self.muy[:n_out], 100*numpy.ones((1,34-n_out)))) 
        else:
            return self.classifier.predict_gmm(x_norm, 10)[0] * self.stdy + self.muy


class MLPPredictor(Predictor):
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
        self.classifier = MLPClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

    def predict(self, x, n_out=34):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        cols = [1] + list(range(3, x.shape[1]))

        x = x[:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, -1)

        if n_out < 34:
            return numpy.hstack((self.classifier.predict(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict(x_norm) * self.stdy + self.muy

class RecurrentMLPPredictor(Predictor):
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
        self.classifier = RecurrentMLP.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

    def predict(self, x):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        cols = [1] + list(range(3, x.shape[1]))

        x = x[:,:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, 1, -1)
        return self.classifier.predict_one(x_norm)[0] * self.stdy + self.muy



class RNNPredictor(Predictor):
    """ Predictor using only a classical RNN network
    (defined as RNNClassifier in classifiers.py)
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
        self.classifier = RNNClassifier.init_from_file(fname)
        self.set_up_means(mux, stdx, muy, stdy)

    def predict(self, x, n_out=34):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 197)

        :return (1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(
            x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))

        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)

        cols = [1] + list(range(3, x.shape[1]))

        x = x[:,:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1,1, -1)

        if n_out < 34:
            return numpy.hstack((self.classifier.predict_one(x_norm) * self.stdy[:n_out] + self.muy[:n_out],  1000*numpy.ones((1,34-n_out))))
        else:
            return self.classifier.predict_one(x_norm) * self.stdy + self.muy
    
    
class UnityMessenger(object):
    """
    Opens a ZMQ socket to communicate with c++ Unity program
    """

    def __init__(self, fname, mux, stdx, muy, stdy, classifier_type, port=5555, n_out=34):
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
        classifier_types = ['Recurrent', 'Classifier', 'MLP', 'RecurrentMLP', 'RNN']

        self.n_out = n_out
        
        # Sets up predictor
        if classifier_type == classifier_types[0]:
            self.predictor = RNNPredictor(fname, mux, stdx, muy, stdy)
            self.recurrent = True
        elif classifier_type == classifier_types[1]:
            self.predictor = FNNPredictor(fname, mux, stdx, muy, stdy)
            self.recurrent = False
        elif classifier_type == classifier_types[2]:
            self.predictor = MLPPredictor(fname, mux, stdx, muy, stdy)
            self.recurrent = False
        elif classifier_type == classifier_types[3]:
            self.predictor = RecurrentMLPPredictor(fname, mux, stdx, muy, stdy)
            self.recurrent = True
        elif classifier_type == classifier_types[4]:
            self.predictor = RNNPredictor(fname, mux, stdx, muy, stdy)
            self.recurrent = True
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

            y = self.predictor.predict(x,n_out=self.n_out).flatten()
#            y[-4:] = 100
            print "Sent Left leg: {0}".format(y[6:10])
            print "max value sent: {0} min value sent: {1}".format(numpy.max(y[6:10]), numpy.min(y[6:10]))
            #  Send reply back to client
            self.socket.send(bytes(str(y)[1:-1]))

if __name__ == '__main__':
    port = 5555
    fname = 'network_output/LSTM_rnn_hidden_[100,30]_epoch_530.json'
    n_out = 30
    classifier_type = 'RNN'

#    x_info = numpy.genfromtxt('mux_stdx_n_13_n_impulse_2000_5.csv', delimiter=',')
#    y_info = numpy.genfromtxt('muy_stdy_n_13_n_impulse_2000_5.csv', delimiter=',')
    x_info = numpy.genfromtxt('mux_stdx_n_13.csv', delimiter=',')
    y_info = numpy.genfromtxt('muy_stdy_n_13.csv', delimiter=',')[:,:-4]

    mux = x_info[0]
    stdx = x_info[1]

    muy = y_info[0]
    stdy = y_info[1]
    messenger = UnityMessenger(
        fname, mux, stdx, muy, stdy, classifier_type, port=port, n_out=n_out)
    messenger.listen()
