from classifiers import RecurrentClassifier
from classifiers import Classifier
import theano
import numpy
import warnings
import zmq


class Predictor(object):

    def __init__(self):
        pass

    def set_up_means(self, mux, stdx, muy, stdy):
        # Set up mean and standard deviation correction
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
        self.muy = muy[:-4]
        self.stdy = stdy[:-4]
        

class RNNPredictor(Predictor):

    def __init__(self, fname, mux, stdx, muy, stdy):
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
        return self.classifier.predict_one(x_norm, 1)[0]*self.stdy + self.muy


class FNNPredictor(Predictor):

    def __init__(self, fname, mux, stdx, muy, stdy):
        self.classifier = Classifier.init_from_file(fname)
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

        x = x[:, cols]
        x_norm = (x - self.mux) * 1. / self.stdx
        x_norm.reshape(1, -1)
        return self.classifier.predict(x_norm, 1)[0]*self.stdy + self.muy


class UnityMessenger(object):
    """
    Opens a ZMQ socket to communicate with c++ Unity program
    """

    def __init__(self, fname, mux, stdx, muy, stdy, recurrent=True, port=5555):
        # Sets up socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:{0}".format(port))
        # Sets up predictor
        if recurrent:
            self.predictor = RNNPredictor(fname, mux, stdx, muy, stdy)
        else:
            self.predictor = FNNPredictor(fname, mux, stdx, muy, stdy)

    def listen(self):
        print "Listening starts"
        while True:
            #  Wait for next request from client
            message = self.socket.recv()
            print "Received request: %s" % message

            x = numpy.fromstring(message, sep=',')
            # Do the work
            if recurrent:
                x = x.reshape(1, 1, -1)
            else:
                x = x.reshape(1, -1)
                
            y = self.predictor.predict(x).flatten()
            print "Sent: {0}".format(y)
            #  Send reply back to client
            self.socket.send(bytes(str(y)[1:-1]))

if __name__ == '__main__':
    port = 5555
    fname = 'network_output/recurrent_gmm_test.json'

    recurrent = True

    x_info = numpy.genfromtxt('mux_stdx_n_13.csv', delimiter=',')
    y_info = numpy.genfromtxt('muy_stdy_n_13.csv', delimiter=',')
        
    mux = x_info[0]
    stdx = x_info[1]

    muy = y_info[0]
    stdy = y_info[1]
    
    messenger = UnityMessenger(fname, mux, stdx, muy, stdy, recurrent=recurrent, port=port)
    messenger.listen()
