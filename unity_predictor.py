from classifiers import RecurrentClassifier
import theano
import numpy
import warnings


class RNNPredictor():
    def __init__(self, fname, mux, stdx):
        self.classifier = RecurrentClassifier.init_from_file(fname)
        if mux.dtype is not theano.config.floatX:
            mux = numpy.asarray(mux, dtype=theano.config.floatX)
            warnings.warn("Mux dtype casted to: {0}".format(theano.config.floatX))
        if stdx.dtype is not theano.config.floatX:
            stdx = numpy.asarray(stdx, dtype=theano.config.floatX)
            warnings.warn("stdx dtype casted to: {0}".format(theano.config.floatX))
        self.mux = mux
        self.stdx = stdx

    def predict(self, x):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 1, 197)

        :return (1,1,30) numpy.array containing the system controls (except the last 4)
        """
        assert type(x) is numpy.ndarray, "Input must be a numpy array. Given type: {0!r}".format(type(x))
        
        if x.dtype is not theano.config.floatX:
            x = numpy.asarray(x, dtype=theano.config.floatX)
        
        cols = [1] + list(range(3,x.shape[2]))
        x = x[:,:, cols]
        x_norm = (x-self.mux)*1./self.stdx
        x_norm.reshape(1,1,-1)
        return self.classifier.predict_one(x_norm, 1)[0]
