from classifiers import RecurrentClassifier
import theano
import numpy
import warnings
import zmq


class RNNPredictor(object):

    def __init__(self, fname, mux, stdx):
        # Loads classifier from fname
        self.classifier = RecurrentClassifier.init_from_file(fname)

        # Stes up mean and standard deviation correction
        if mux.dtype is not theano.config.floatX:
            mux = numpy.asarray(mux, dtype=theano.config.floatX)
            warnings.warn("Mux dtype casted to: {0}".format(
                theano.config.floatX))
        if stdx.dtype is not theano.config.floatX:
            stdx = numpy.asarray(stdx, dtype=theano.config.floatX)
            warnings.warn("stdx dtype casted to: {0}".format(
                theano.config.floatX))
        self.mux = mux
        self.stdx = stdx

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
        return self.classifier.predict_one(x_norm, 1)[0]


class FNNPredictor(object):

    def __init__(self, fname, mux, stdx):
        self.classifier = Classifier.init_from_file(fname)

        if mux.dtype is not theano.config.floatX:
            mux = numpy.asarray(mux, dtype=theano.config.floatX)
            warnings.warn("Mux dtype casted to: {0}".format(
                theano.config.floatX))
        if stdx.dtype is not theano.config.floatX:
            stdx = numpy.asarray(stdx, dtype=theano.config.floatX)
            warnings.warn("stdx dtype casted to: {0}".format(
                theano.config.floatX))
        self.mux = mux
        self.stdx = stdx

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

        cols = [1] + list(range(3, x.shape[1]))
        x = x[:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, -1)
        return self.classifier.predict(x_norm, 1)[0]


class UnityMessenger(object):
    """
    Opens a ZMQ socket to communicate with c++ Unity program
    """

    def __init__(self, fname, mux, stdx, port=5555):
        # Sets up socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))
        # Sets up predictor
        self.predictor = RNNPredictor(fname, mux, stdx)

    def listen(self):
        print "Listening starts"
        while True:
            #  Wait for next request from client
            message = self.socket.recv()
            print "Received request: %s" % message

            # Do the work
            x = numpy.fromstring(message, sep=',').reshape(1, 1, -1)
            y = self.predictor.predict(x).flatten()
            #  Send reply back to client
            self.socket.send(bytes(str(y)[1:-1]))

if __name__ == '__main__':
    port = 5555
    fname = 'network_output/recurrentclassifier_LSTM_n_13_mlp_hidden_[10]_mlp_activation_[sigmoid]_lbn_n_hidden_[150,100,50]_det_activations_[linear,linear,linear,linear]_stoch_activations_[sigmoid,sigmoid]_m_10_bsize_100_method_SGD/networks/recurrentclassifier_LSTM_n_13_mlp_hidden_[10]_mlp_activation_[sigmoid]_lbn_n_hidden_[150,100,50]_det_activations_[linear,linear,linear,linear]_stoch_activations_[sigmoid,sigmoid]_m_10_bsize_100_method_SGD_epoch_3000.json'
    mux = numpy.array(0)
    stdx = numpy.array(1)
    messenger = UnityMessenger(fname, mux, stdx, port=port)
    messenger.listen()
