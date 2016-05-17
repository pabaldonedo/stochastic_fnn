import theano.tensor as T
import theano
import numpy as np
import logging
from types import IntType
from types import ListType
from types import FloatType
import json
import os
from optimizer import SGD
from optimizer import RMSProp
from optimizer import AdaDelta
from optimizer import AdaGrad
from optimizer import Adam
from mlp import MLPLayer
from lbn import LBN
from LBNRNN import LBNRNN_module
from util import load_states
from util import load_controls
from util import log_init
from util import flatten


class Classifier(object):

    def parse_inputs(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                                        det_activations, stoch_activations, stoch_n_hidden, log):
        self.log = log
        if self.log is None:
            logging.basicConfig(level=logging.INFO)
            self.log = logging.getLogger()


        assert type(n_in) is IntType, "n_in must be an integer: {0!r}".format(n_in)
        assert type(mlp_n_in) is IntType, "nlp_n_in must be an integer: {0!r}".format(mlp_n_in)

        assert type(n_out) is IntType, "n_out must be an integer: {0!r}".format(n_out)
        assert type(mlp_n_hidden) is ListType, "mlp_n_hidden must be a list: {0!r}".\
                                                                        format(mlp_n_hidden)
        assert type(mlp_activation_names) is ListType, "mlp_activation_names must be a list:"\
                                                                        " {0!r}".\
                                                                        format(mlp_n_hidden)
        assert type(lbn_n_hidden) is ListType, "lbn_n_hidden must be a list: {0!r}".\
                                                                        format(lbn_n_hidden)                                                                
        assert type(det_activations) is ListType, "det_activations must be a list: {0!r}".\
                                                                        format(det_activations)
        assert type(stoch_activations) is ListType, "stoch_activations must be a list: {0!r}".\
                                                                        format(stoch_activations)
        assert type(stoch_n_hidden) is ListType, "stoch_n_hidden must be a list: {0!r}".\
                                                                        format(stoch_n_hidden)

        self.lbn_n_hidden = lbn_n_hidden
        self.det_activations = det_activations
        self.stoch_activations = stoch_activations
        self.n_in = n_in
        self.stoch_n_hidden = stoch_n_hidden
        self.n_out = n_out

    def set_up_mlp(self, mlp_n_hidden, mlp_activation_names, mlp_n_in, weights, timeseries_layer=False):
        self.mlp_n_hidden = mlp_n_hidden
        self.bone_representations = [None]*15
        self.mlp_activation_names = mlp_activation_names
        self.mlp_n_in = mlp_n_in
        for i in xrange(len(self.bone_representations)):
            if i == 0:
                bone_mlp = MLPLayer(mlp_n_in-2, self.mlp_n_hidden, self.mlp_activation_names,
                                            input_var = self.x[:,:,:mlp_n_in-2] if timeseries_layer
                                                                        else self.x[:,:mlp_n_in-2],
                                            layers_info = None if weights is
                                            None else weights['bone_mlps'][i]['MLPLayer'],
                                            timeseries_network=timeseries_layer)
            else:
                bone_mlp = MLPLayer(mlp_n_in, self.mlp_n_hidden, self.mlp_activation_names,
                                            input_var = self.x[:,:,i*mlp_n_in-2:(i+1)*mlp_n_in-2]
                                            if timeseries_layer else
                                                            self.x[:,i*mlp_n_in-2:(i+1)*mlp_n_in-2],
                                            layers_info = None if weights is None else
                                                                weights['bone_mlps'][i]['MLPLayer'],
                                            timeseries_network=timeseries_layer)
            self.bone_representations[i] = bone_mlp


    def __init__(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                 det_activations, stoch_activations, stoch_n_hidden=[-1], log=None, weights=None):

        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.parse_inputs(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                                        det_activations, stoch_activations, stoch_n_hidden, log)

        self.set_up_mlp(mlp_n_hidden, mlp_activation_names, mlp_n_in, weights)

        self.lbn_input = T.concatenate([bone.output for bone in self.bone_representations] +
                                                                            [self.x[:,-2:]], axis=1)

        self.lbn = LBN(len(self.bone_representations)*self.mlp_n_hidden[-1]+2, self.lbn_n_hidden,
                                                        self.n_out,
                                                        self.det_activations,
                                                        self.stoch_activations,
                                                        input_var=self.lbn_input,
                                                        layers_info=None if weights is None else
                                                                        weights['lbn']['layers'])

        self.y = self.lbn.y
        self.m = self.lbn.m
        self.get_log_likelihood = theano.function(inputs=[self.x, self.lbn.y, self.lbn.m],
                                                outputs=self.lbn.log_likelihood)

        
        mlp_params = [mlp_i.params for mlp_i in self.bone_representations]
        self.params = [mlp_params, self.lbn.params]
        self.predict = theano.function(inputs=[self.x, self.lbn.m], outputs=self.lbn.output)
        self.log.info("Network created with n_in: {0}, mlp_n_hidden: {1}, "
                        "mlp_activation_names: {2}, lbn_n_hidden: {3}, det_activations: {4}, "
                        "stoch_activations: {5}, n_out: {6}".format(
                        self.n_in, self.mlp_n_hidden, self.mlp_activation_names, self.lbn_n_hidden,
                        self.det_activations, self.stoch_activations, self.n_out))

    def save_network(self, fname):
        output_string = self.generate_saving_string()
        with open(fname, 'w') as f:
            f.write(output_string)
        self.log.info("Network saved.")

    def generate_saving_string(self):

        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                    "mlp_n_in": self.mlp_n_in, "mlp_n_hidden": self.mlp_n_hidden,
                                    "mlp_activation_names": self.mlp_activation_names, 
                                    "lbn_n_hidden": self.lbn_n_hidden,
                                    "det_activations": self.det_activations,
                                    "stoch_activations": self.stoch_activations})
        output_string += ",\"layers\": {\"bone_mlps\":["
        for i, bone in enumerate(self.bone_representations):
            if i > 0:
                output_string += ","
            output_string += "{\"MLPLayer\":"
            output_string += bone.generate_saving_string()
            output_string += "}"
        output_string += "]"
        output_string += ",\"lbn\":"
        output_string += self.lbn.generate_saving_string()
        output_string += "}}"

        return output_string

    def get_call_back(self, save_every, fname, epoch0):
        c = callBack(self, save_every, fname, epoch0)
        return c.cback

    def get_cost(self):
        cost = -1./self.x.shape[0]*self.lbn.log_likelihood
        return cost

    def fit(self, x, y, m, n_epochs, b_size, method, save_every=1, fname=None, epoch0=1,
                                                                        x_test=None, y_test=None):

        l = self.get_log_likelihood(x,y,m)
        self.log.info("log_likelihood {0} and mean: {1}".format(l, l*1./x.shape[0]))
        self.log.info("Number of samples: {0}:".format(x.shape[0]))
        train_set_x = theano.shared(np.asarray(x,
                                        dtype=theano.config.floatX))

        train_set_y = theano.shared(np.asarray(y,
                                        dtype=theano.config.floatX))

        if x_test is not None and y_test is not None:
            test_set_x = theano.shared(np.asarray(x_test,
                                            dtype=theano.config.floatX))

            test_set_y = theano.shared(np.asarray(y_test,
                                            dtype=theano.config.floatX))
        
        flat_params = flatten(self.params)
        cost = self.get_cost()
        compute_error = theano.function(inputs=[self.x, self.y], outputs=cost,
                                        givens={self.m: m})
       
        allowed_methods = ['SGD', "RMSProp", "AdaDelta", "AdaGrad", "Adam"]

        if method['type'] == allowed_methods[0]:
            opt = SGD(method['lr_decay_schedule'], method['lr_decay_parameters'],
                    method['momentum_type'], momentum=method['momentum'])
        elif method['type'] == allowed_methods[1]:
            opt = RMSProp(method['learning_rate'], method['rho'], method['epsilon'])
        elif method['type'] == allowed_methods[2]:
            opt = AdaDelta(method['learning_rate'], method['rho'], method['epsilon'])
        elif method['type'] == allowed_methods[3]:
            opt = AdaGrad(method['learning_rate'], method['epsilon'])
        elif method['type'] == allowed_methods[4]:
            opt = Adam(method['learning_rate'], method['b1'], method['b2'], method['e'])
        else:
            raise NotImplementedError, \
                "Optimization method not implemented. Choose one out of: {0}".format(
                                                                                    allowed_methods)

        self.log.info("Fit starts with epochs: {0}, batch size: {1}, method: {2}".format(
                                                                        n_epochs, b_size, method))

        opt.fit(self.x, self.y, train_set_x, train_set_y, b_size, cost, flat_params, n_epochs,
                                    compute_error, self.get_call_back(save_every, fname, epoch0),
                                    extra_train_givens={self.m:m},
                                    test_set_x=test_set_x, test_set_y=test_set_y)

        """
        self.fiting_variables(b_size, train_set_x, train_set_y)


        gparams = [T.grad(cost, p) for p in flat_params]
        v = [theano.shared(value=np.zeros(th.shape.eval(), dtype=theano.config.floatX)) for th in flat_params]
        v_upds = [method['momentum']*vi - method['learning_rate']*gp for vi,gp in zip(v, gparams)]
        upd = [(vi, v_updi) for vi, v_updi in zip(v, v_upds)]
        upd += [(p, p-method['learning_rate']*gp+method['momentum']*v_upd) for p, gp, v_upd in zip(flat_params, gparams, v_upds)]

        train_model = theano.function(inputs=[self.index, self.n_ex],
                                    outputs=self.lbn.log_likelihood,
                                    updates=upd,
                                    givens={self.x:train_set_x[self.batch_start:self.batch_stop],
                                            self.lbn.y:train_set_y[self.batch_start:self.batch_stop],
                                            self.lbn.m: m})

        epoch = 0
        while epoch < n_epochs:
            for minibatch_idx in xrange(self.n_train_batches):
                minibatch_avg_cost = train_model(minibatch_idx, self.n_train)

            train_error = compute_error(x,y)
            log_likelihood = self.get_log_likelihood(x,y,m)
            self.log.info("epoch: {0} train_error: {1}, log_likelihood: {2} with".format(
                                                                    epoch+epoch0, train_error,
                                                                    log_likelihood))

        """
    def fiting_variables(self, batch_size, train_set_x, train_set_y, test_set_x=None):
        """Sets useful variables for locating batches"""    
        self.index = T.lscalar('index')    # index to a [mini]batch
        self.n_ex = T.lscalar('n_ex')      # total number of examples

        assert type(batch_size) is IntType or FloatType, "Batch size must be an integer."
        if type(batch_size) is FloatType:
            warnings.warn('Provided batch_size is FloatType, value has been truncated')
            batch_size = int(batch_size)
        # Proper implementation of variable-batch size evaluation
        # Note that the last batch may be a smaller size
        # So we keep around the effective_batch_size (whose last element may
        # be smaller than the rest)
        # And weight the reported error by the batch_size when we average
        # Also, by keeping batch_start and batch_stop as symbolic variables,
        # we make the theano function easier to read
        self.batch_start = self.index * batch_size
        self.batch_stop = T.minimum(self.n_ex, (self.index + 1) * batch_size)
        self.effective_batch_size = self.batch_stop - self.batch_start

        self.get_batch_size = theano.function(inputs=[self.index, self.n_ex],
                                          outputs=self.effective_batch_size)

        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        self.n_train = train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches = int(np.ceil(1.0 * self.n_train / batch_size))
        if test_set_x is not None:
            self.n_test = test_set_x.get_value(borrow=True).shape[0]
            self.n_test_batches = int(np.ceil(1.0 * self.n_test / batch_size))

    @classmethod
    def init_from_file(cls, fname, log=None):
        with open(fname, 'r') as f:
            network_description = json.load(f)

        network_properties = network_description['network_properties']
        loaded_classifier = cls(network_properties['n_in'],
                                network_properties['n_out'],
                                network_properties['mlp_n_in'],
                                network_properties['mlp_n_hidden'],
                                network_properties['mlp_activation_names'],
                                network_properties['lbn_n_hidden'],
                                network_properties['det_activations'],
                                network_properties['stoch_activations'],
                                log=log,
                                weights=network_description['layers'])

        return loaded_classifier



class RecurrentClassifier(Classifier):
            
    def parse_inputs(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                     lbn_n_out, det_activations, stoch_activations, stoch_n_hidden,
                     rnn_hidden, rnn_activations, log):
        super(RecurrentClassifier, self).parse_inputs(n_in, n_out, mlp_n_in, mlp_n_hidden,
                                                      mlp_activation_names, lbn_n_hidden,
                                                      det_activations, stoch_activations,
                                                      stoch_n_hidden, log)
        
        assert type(lbn_n_out) is IntType, "lbn_n_out must be an integer: {0!r}".format(lbn_n_out)
        assert type(rnn_hidden) is ListType, "rnn_hidden must be a list: {0!r}".format(rnn_hidden)
        assert type(rnn_activations) is ListType, "rnn_activations must be a list: {0!r}".format(
                                                                                rnn_activations)
        self.lbn_n_out = lbn_n_out
        self.rnn_hidden = rnn_hidden
        self.rnn_activations = rnn_activations


    def __init__(self, n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                                    lbn_n_out, det_activations, stoch_activations,
                                    rnn_hidden, rnn_activations, stoch_n_hidden=[-1],
                                    log=None, weights=None):

        self.x = T.tensor3('x', dtype=theano.config.floatX)
        self.parse_inputs(n_in, n_out, mlp_n_in,
                          mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                          lbn_n_out, det_activations, stoch_activations, stoch_n_hidden,
                          rnn_hidden, rnn_activations, log)
        self.set_up_mlp(mlp_n_hidden, mlp_activation_names, mlp_n_in, weights, timeseries_layer=True)

        self.lbn_input = T.concatenate([bone.output for bone in self.bone_representations] +
                                                                            [self.x[:,:,-2:]], axis=2)
        

        lbn_properties = {'n_in':len(self.bone_representations)*self.mlp_n_hidden[-1]+2,
                        'n_hidden':self.lbn_n_hidden, 'n_out':lbn_n_out,
                        'det_activations':self.det_activations,
                        'stoch_activations':self.stoch_activations,
                        'stoch_n_hidden': self.stoch_n_hidden,
                        'input_var':self.lbn_input,
                        'layers': None if weights is None else weights['lbnrnn']['lbn']['layers']}

        rnn_properties = {'n_in': lbn_properties['n_out'],
                        'n_out': self.n_out,
                        'n_hidden': self.rnn_hidden,
                        'activations': self.rnn_activations,
                        'layers': None if weights is None else weights['lbnrnn']['rnn']['layers']}

        self.lbnrnn = LBNRNN_module(lbn_properties, rnn_properties, input_var=self.lbn_input)

        self.y = self.lbnrnn.y
        self.m = self.lbnrnn.lbn.m
        mlp_params = [mlp_i.params for mlp_i in self.bone_representations]
        self.params = [mlp_params, self.lbnrnn.params]
        self.get_log_likelihood = theano.function(inputs=[self.x, self.lbnrnn.y, self.lbnrnn.lbn.m],
                                                outputs=self.lbnrnn.log_likelihood)

        self.output = self.lbnrnn.output
        self.predict = theano.function(inputs=[self.x, self.lbnrnn.lbn.m], outputs=self.output)
        self.log.info("Network created with n_in: {0}, mlp_n_hidden: {1}, "
                        "mlp_activation_names: {2}, lbn_n_hidden: {3}, det_activations: {4}, "
                        "stoch_activations: {5}, n_out: {6}".format(
                        self.n_in, self.mlp_n_hidden, self.mlp_activation_names, self.lbn_n_hidden,
                        self.det_activations, self.stoch_activations, self.n_out))

    def get_cost(self):
        cost = -1./(self.x.shape[0]*self.x.shape[1])*self.lbnrnn.log_likelihood

        return cost


    def generate_saving_string(self):
        output_string = "{\"network_properties\":"
        output_string += json.dumps({"n_in": self.n_in, "n_out": self.n_out,
                                    "mlp_n_in": self.mlp_n_in, "mlp_n_hidden": self.mlp_n_hidden,
                                    "mlp_activation_names": self.mlp_activation_names, 
                                    "lbn_n_hidden": self.lbn_n_hidden,
                                    "lbn_n_out": self.lbn_n_out,
                                    "det_activations": self.det_activations,
                                    "stoch_activations": self.stoch_activations,
                                    "rnn_hidden": self.rnn_hidden,
                                    "rnn_activations": self.rnn_activations })
        output_string += ",\"layers\": {\"bone_mlps\":["
        for i, bone in enumerate(self.bone_representations):
            if i > 0:
                output_string += ","
            output_string += "{\"MLPLayer\":"
            output_string += bone.generate_saving_string()
            output_string += "}"
        output_string += "]"
        output_string += ",\"lbnrnn\":"
        output_string += self.lbnrnn.generate_saving_string()
        output_string += "}}"

        return output_string


class callBack:
    def __init__(self, classifier, save_every, fname, epoch0):


        self.epoch0 = epoch0
        self.log_likelihoods = []
        self.epochs = []
        self.classifier = classifier

        opath = os.path.dirname(fname)
        file_name = os.path.basename(fname)
        like_file = '{0}/likelihoods/{1}.csv'.format(opath, file_name)

        self.likelihood_file(like_file)
        self.save_every = save_every

        network_name = '{0}/networks/{1}'.format(opath, file_name)
        if not os.path.exists('{0}/networks'.format(opath)):
            os.makedirs('{0}/networks'.format(opath))
        self.fname = network_name

    def likelihood_file(self, fname):
        path_fname = os.path.dirname(fname)
        if not os.path.exists(path_fname):
            os.makedirs(path_fname)

        def save_likelihood(epochs, log_likelihoods, test_like=None):
            with open(fname, 'a') as f:
                for e, l in zip(epochs, log_likelihoods):
                    f.write('{0},{1}\n'.format(e, l))
            if test_like is not None:
                test_fname = os.path.splitext(os.path.basename(fname))
                with open('{0}_test.csv'.format(test_fname), 'a') as f:
                    for e, l in zip(epochs, test_like):
                        f.write('{0},{1}\n'.format(e, l))                
            self.classifier.log.info("Log likelihoods saved.")

        self.save_likelihood = save_likelihood

    def cback(self, epoch, n_samples, train_error=None, opt_parameters=None, test_error=None):
        log_likelihood = -n_samples*train_error
        if test_error is None:
            self.classifier.log.info("epoch: {0} train_error: {1}, log_likelihood: {2} with" \
                                "options: {3}.".format(epoch+self.epoch0, train_error,
                                                                    log_likelihood, opt_parameters))
        else:
            self.classifier.log.info("epoch: {0} train_error: {1}, test_error: {2} "\
                                "log_likelihood: {3} with options: {4}.".format(
                                                                    epoch+self.epoch0, train_error,
                                                                    test_error,
                                                                    log_likelihood, opt_parameters))


        self.epochs.append(epoch+self.epoch0)
        self.log_likelihoods.append(log_likelihood)
        if epoch % self.save_every == 0 and self.fname is not None:
            self.classifier.save_network("{0}_epoch_{1}.json".format(self.fname, epoch+self.epoch0))
            self.save_likelihood(self.epochs, self.log_likelihoods)
            self.log_likelihoods = []
            self.epochs = []

    
def main():
    n = 7
    recurrent = False
    seq_len = 61
    train_size = 0.8
    x = load_states(n)
    mux = np.mean(x,axis=0)
    stdx = np.std(x,axis=0)
    stdx[stdx==0] = 1.
    x = (x-mux)*1./stdx
    if recurrent:
        x = x.reshape(seq_len, -1, x.shape[1])

        idx = np.random.permutation(x.shape[1])
        cols = [1]+list(range(3,x.shape[2]))
        x = x[:,:,cols]
        x = x[:,idx,:]
        train_bucket = int(np.ceil(x.shape[1]*train_size))
        x_train = x[:,:train_bucket]
        x_test = x[:,train_bucket:]

    else:
        idx = np.random.permutation(x.shape[0])
        cols = [1]+list(range(3,x.shape[1]))
        x = x[:,cols]
        x = x[idx]
        train_bucket = int(np.ceil(x.shape[0]*train_size))
        x_train = x[:train_bucket]
        x_test = x[train_bucket:]

    y = load_controls(n)
    muy = np.mean(y,axis=0)
    stdy = np.std(y,axis=0)
    stdy[stdy==0] = 1.
    y = (y-muy)*1./stdy
    if recurrent:
        y = y.reshape(seq_len, -1, y.shape[1])
        y = y[:,idx,:-4]
        y_train = y[:train_bucket]
        y_test = y[train_bucket:]
        n_in = x.shape[2]
        n_out = y.shape[2]
    else:
        y = y[idx,:-4]
        y_train = y[:train_bucket]
        y_test = y[train_bucket:]
        n_in = x.shape[1]
        n_out = y.shape[1]

    mlp_activation_names = ['sigmoid']
    lbn_n_hidden = [150, 100, 50]
    det_activations = ['linear', 'linear','linear', 'linear']
    stoch_activations = ['sigmoid', 'sigmoid']
    
    mlp_n_in = 13
    mlp_n_hidden = [10]
    b_size= 100
    n_epochs = 5
    lr = .05
    save_every = 1
    m = 10
    opt_type = 'SGD'

    method={'type':opt_type, 'lr_decay_schedule':'constant', 'lr_decay_parameters':[lr],
            'momentum_type': 'nesterov', 'momentum': 0.01, 'b1': 0.9, 'b2':0.999, 'e':1e-6,
            'learning_rate':lr}
    epoch0 = 1
    rnn_hidden = [50]
    rnn_activations = ['sigmoid', 'linear']
    lbn_n_out = 50
    network_name = "{0}_n_{1}_mlp_hidden_[{2}]_mlp_activation_[{3}]_lbn_n_hidden_[{4}]"\
                    "_det_activations_[{5}]_stoch_activations_[{6}]_m_{7}_bsize_{8}_method_{9}".\
                                                    format(
                                                    'recurrentclassifier' if recurrent else 
                                                                                    'classifier',
                                                    n,
                                                    ','.join(str(e) for e in mlp_n_hidden),
                                                    ','.join(str(e) for e in mlp_activation_names),
                                                    ','.join(str(e) for e in lbn_n_hidden),
                                                    ','.join(str(e) for e in det_activations),
                                                    ','.join(str(e) for e in stoch_activations),
                                                    m, b_size, method['type'])

    opath = "network_output/{0}".format(network_name)
    if not os.path.exists(opath):
        os.makedirs(opath)
    fname = '{0}/{1}.json'.format(opath, network_name)
    network_fname = '{0}/networks/{1}'.format(opath, network_name)
    log, session_name = log_init(opath)#, session_name='donette')
    if recurrent:
        c = RecurrentClassifier(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names,
                                lbn_n_hidden,
                                lbn_n_out, det_activations, stoch_activations,
                                rnn_hidden, rnn_activations)

    else: 
    #c = Classifier.init_from_file('{0}_epoch_{1}'.format(network_fname, epoch0-1), log=log)

        c = Classifier(n_in, n_out, mlp_n_in, mlp_n_hidden, mlp_activation_names, lbn_n_hidden,
                                                            det_activations,
                                                            stoch_activations, log=log)



    f = c.fit(x,y,m,n_epochs, b_size, method, fname=fname, epoch0=epoch0,
                                                                    x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
