from classifiers import RecurrentClassifier

class RNNPredictor():
    def __init__(self, fname, mux, stdx):
        self.classifier = RecurrentClassifier.init_from_file(fname)

    def predict(self, x):
        """
        :type x: numpy.array
        :param x: input data of shape (1, 1, 197)

        :return (1,1,30) numpy.array containing the system controls (except the last 4)
        """
        cols = [1] + list(range(3,x.shape[2]))
        x = x[:,:, cols]
        x_norm = (x-mux)*1./stdx
        x_norm.reshape(1,1,-1)
        return classifier.predict(x_norm, 1)

