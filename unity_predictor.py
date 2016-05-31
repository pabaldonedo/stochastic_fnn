from classifiers import RecurrentClassifier

class RNNPredictor():
    def __init__(self, fname, mux, stdx):
        self.classifier = RecurrentClassifier.init_from_file(fname)
        self.mux = mux
        self.stdx = stdx

    def predict(self, x):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 1, 197)

        :return (1,1,30) numpy.array containing the system controls (except the last 4)
        """
        cols = [1] + list(range(3,x.shape[2]))
        x = x[:,:, cols]
        x_norm = (x-self.mux)*1./self.stdx
        x_norm.reshape(1,1,-1)
        return classifier.predict(x_norm, 1)

