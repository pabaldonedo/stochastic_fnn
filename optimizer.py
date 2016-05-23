import warnings
import types
import scipy
import theano
import theano.tensor as T
from types import FloatType
from types import IntType
from types import ListType
from types import TupleType
import numpy as np
import logging
from collections import OrderedDict


class Optimizer():

    def fiting_variables(self, batch_size, train_set_x, test_set_x=None, sample_axis=0):
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


    def fit(self):
        """To be implemented in subclasses. Performs the optimization."""
        raise NotImplementedError

class GradientBased(Optimizer):

    def get_updates(self, theta, cost=None, gtheta=None):
        raise NotImplementedError

    def fit(self, x, y, x_train, y_train, batch_size, cost, theta, n_epochs,
                            compute_error, call_back, x_test=None, y_test=None,
                            validate_every=1, extra_train_givens={}, chunk_size=None,
                            sample_axis=0):
        """Performs the optimization using a Gradient Based algorithm.

        :param x: theano input variable of the rnn.
        :param y: theano output variable of the rnn.
        :param batch_size: integer #samples used per batch.
        :param cost: theano variable equals to the cost of the rnn.
        :param theta: List containing the parametsr.
        :param n_epochs: integer number of epochs to train.
        :param compute_error: theano function that computes error given predicted output and true output.
        :param call_back: call back function to call for printing information.
        :param test_set_x: test set matrix.
                            Dimensions: sequence length x #training samples x input dimensionality. 
        :param validate_every: telling every how many epochs the fit functions reports to the
                               call_back function. It can be a float.
        """
        data_shape = x_train.shape
        seq_len = 1 if len(data_shape) == 2 else np.prod(data_shape[:-2])

        if chunk_size is None:
            n_chunks = 1
            chunk_size = x_train.shape[sample_axis]


            train_set_x = theano.shared(np.asarray(x_train,
                                        dtype=theano.config.floatX))

            train_set_y = theano.shared(np.asarray(y_train,
                                        dtype=theano.config.floatX))
        
        else:    
            n_chunks = int(np.ceil(x_train.shape[sample_axis]*1./chunk_size))
            x_train = np.asarray(x_train, dtype=theano.config.floatX)
            y_train = np.asarray(y_train, dtype=theano.config.floatX)
            train_set_x = theano.shared(x_train[:chunk_size])
            train_set_y = theano.shared(y_train[:chunk_size])
        
        if x_test is not None and y_test is not None:
            
            test_set_x = theano.shared(np.asarray(x_test[:chunk_size],
                                            dtype=theano.config.floatX))

            test_set_y = theano.shared(np.asarray(y_test[:chunk_size],
                                            dtype=theano.config.floatX))
        
            test_n_chunks = int(np.ceil(x_test.shape[sample_axis]*1./chunk_size))
        else:
            test_set_x = None
            test_set_y = None
        self.test_availavility = test_set_x is not None
        #Setting up indicator variables for looping along batches
        
        self.fiting_variables(batch_size, train_set_x, test_set_x=test_set_x,
                              sample_axis=sample_axis)
        self.n_train = x_train.shape[sample_axis]
        self.n_test = x_test.shape[sample_axis]
        gtheta = [None]*len(theta)
        updates = []
        self.it = theano.shared(np.asarray(1., dtype=theano.config.floatX))
        i_t = self.it +np.asarray(1., dtype=theano.config.floatX)
        updates.append((self.it, i_t))
        for i, th in enumerate(theta):
            gtheta[i] = T.grad(cost, th)
            updates += self.get_updates(th, gtheta=gtheta[i])
        
        
        #Variables to keep track of the error while training
        train_log_likelihood_evolution = []
        test_log_likelihood_evolution = []

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        if sample_axis == 0:
            givens_default_values = {x: train_set_x[self.batch_start:self.batch_stop],
                                     y: train_set_y[self.batch_start:self.batch_stop]}
        elif sample_axis == 1:
            givens_default_values = {x: train_set_x[:,self.batch_start:self.batch_stop],
                                     y: train_set_y[:,self.batch_start:self.batch_stop]}
        else:
            raise NotImplementedError

        givens_default_values.update(extra_train_givens)

        outputs = [cost, self.batch_stop-self.batch_start]
        if 'lr' in self.opt_parameters.keys():
            outputs = [cost, self.batch_stop-self.batch_start, self.opt_parameters['lr']]

        train_model = theano.function(inputs=[self.index, self.n_ex],
            outputs=outputs,
            updates=updates,
            givens=givens_default_values,
            on_unused_input='warn')
        epoch = 0

        while epoch < n_epochs:
            data_log_likelihood = 0
            for chunk in xrange(n_chunks):
                this_chunk_size = train_set_x.get_value().shape[sample_axis]
                n_train_batches = int(np.ceil(1.0 * this_chunk_size / batch_size))
                tmp = 0
                for minibatch_idx in xrange(n_train_batches):
                   
                    if 'lr' in self.opt_parameters.keys():
                        minibatch_avg_cost, this_batch_size, l_r = train_model(minibatch_idx, this_chunk_size)
                        self.opt_parameters['lr'] = l_r
                    else: 
                        minibatch_avg_cost, this_batch_size = train_model(minibatch_idx, this_chunk_size)

                    data_log_likelihood -= minibatch_avg_cost*this_batch_size*seq_len
                    tmp +=this_batch_size
                print "TOTAL BATCH: {0}, CHUNK SIZE: {1}".format(tmp, this_chunk_size)
                if chunk < n_chunks - 1:
                    #train_set_x.set_value(x_train[(chunk+1)*chunk_size:min((chunk+2)*chunk_size, x_train.shape[0])])
                    train_set_x.set_value(x_train.take(xrange((chunk+1)*chunk_size,min((chunk+2)*chunk_size, x_train.shape[sample_axis])), axis=sample_axis))
                    #train_set_y.set_value(y_train[(chunk+1)*chunk_size:min((chunk+2)*chunk_size, y_train.shape[0])])
                    train_set_y.set_value(y_train.take(xrange((chunk+1)*chunk_size,min((chunk+2)*chunk_size, y_train.shape[sample_axis])), axis=sample_axis))
          
            train_log_likelihood_evolution.append((epoch, data_log_likelihood))
            
            if self.test_availavility:
                test_log_likelihood = 0
                for chunk in xrange(test_n_chunks):
                    this_chunk_size = test_set_x.get_value().shape[sample_axis]
                    test_log_likelihood -= compute_error(test_set_x.eval(), test_set_y.eval())*this_chunk_size*seq_len
                    
                    if chunk < test_n_chunks - 1:
                        #test_set_x.set_value(x_test[(chunk+1)*chunk_size:min((chunk+2)*chunk_size, x_test.shape[0])])
                        test_set_x.set_value(x_test.take(xrange((chunk+1)*chunk_size,min((chunk+2)*chunk_size, x_test.shape[sample_axis])), axis=sample_axis))
                        #test_set_y.set_value(y_test[(chunk+1)*chunk_size:min((chunk+2)*chunk_size, y_test.shape[0])])
                        test_set_y.set_value(y_test.take(xrange((chunk+1)*chunk_size,min((chunk+2)*chunk_size, y_test.shape[sample_axis])), axis=sample_axis))
                test_log_likelihood_evolution.append((epoch, test_log_likelihood))

                call_back(epoch, self.n_train, train_log_likelihood=data_log_likelihood,
                                                                opt_parameters=self.opt_parameters,
                                                                test_log_likelihood=test_log_likelihood,
                                                                n_test=self.n_test)
            else:
                call_back(epoch, self.n_train, train_log_likelihood=data_log_likelihood,
                                                                opt_parameters=self.opt_parameters)
            epoch = epoch + 1

        return train_log_likelihood_evolution, test_log_likelihood_evolution


class RMSProp(GradientBased):

    def __init__(self, learning_rate, rho, epsilon):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.rho = rho
        assert type(self.rho) is FloatType or IntType, "Rho decay must be an integer or float: {0!r}".format(self.rho)
        assert 0 < self.rho, "Rho decay must be positive: {0!r}".format(self.rho)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.opt_parameters = {'opt': 'RMSProp', 'lr':self.learning_rate, 'rho':self.rho,'e':self.epsilon}

        logging.info('Optimizer loaded. Type: {0}, learning rate: {1}, rho decay: {2},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                self.opt_parameters['rho'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        updates = []
        rms = theano.shared(theta.get_value() *0.)
        rms_upd = self.rho*rms + (1 - self.rho) * T.sqr(gtheta)
        theta_upd = theta - self.learning_rate * gtheta / T.sqrt(rms_upd + self.epsilon)

        updates.append((rms, rms_upd))
        updates.append((theta, theta_upd))
        return updates


class AdaDelta(GradientBased):

    def __init__(self, learning_rate, rho, epsilon):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.rho = rho
        assert type(self.rho) is FloatType or IntType, "Rho decay must be an integer or float: {0!r}".format(self.rho)
        assert 0 < self.rho, "Rho decay must be positive: {0!r}".format(self.rho)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.opt_parameters = {'opt': 'AdaDelta', 'lr':self.learning_rate, 'rho':self.rho,'e':self.epsilon}
        logging.info('Optimizer loaded. Type: {0}, learning rate: {1}, rho decay: {2},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                self.opt_parameters['rho'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        updates = []

        eg2 = theano.shared(theta.get_value() * 0.) 
        edth2 = theano.shared(theta.get_value() * 0.)

        eg2_upd = self.rho*eg2 + (1-self.rho)*T.sqr(gtheta)

        rms_dth_tm1 = T.sqrt(edth2 + self.epsilon)
        rms_gtheta_t = T.sqrt(eg2_upd + self.epsilon)
        dth = - gtheta * rms_dth_tm1/ rms_gtheta_t
        edth2_upd = self.rho*edth2 + (1-self.rho)*dth**2
        theta_upd = theta + self.learning_rate*dth

        updates.append((eg2, eg2_upd))
        updates.append((edth2, edth2_upd))
        updates.append((theta, theta_upd))
        return updates


class AdaGrad(GradientBased):

    def __init__(self, learning_rate, epsilon):
        self.learning_rate = learning_rate
        assert type(self.learning_rate) is FloatType or IntType, "Learning rate must be an integer or float: {0!r}".format(self.learning_rate)
        assert 0 < self.learning_rate, "Learning rate must be positive: {0!r}".format(self.learning_rate)
        self.epsilon = epsilon
        assert type(self.epsilon) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.epsilon)
        assert 0 < self.epsilon, "Epsilon must be positive: {0!r}".format(self.epsilon)  
        self.opt_parameters = {'opt': 'AdaGrad', 'learning_rate':self.learning_rate, 'epsilon':self.epsilon}
        logging.info('Optimizer loaded. Type: {0}, learning rate: {1},'
            ' epsilon: {3}'.format(self.opt_parameters['opt'], self.opt_parameters['lr'],
                                    self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        updates = []
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        g2_sum = theano.shared(theta.get_value() * 0.)
        g2_sum_upd = g2_sum + T.sqr(gtheta) 
        theta_upd = theta - self.learning_rate * gtheta/T.sqrt(g2_sum_upd + self.epsilon) 
        
        updates.append((g2_sum, g2_sum_upd))
        updates.append((theta, theta_upd))
        return updates

class Adam(GradientBased):
    def __init__(self, step_size, b1, b2, e):
        """:param  step_size. positive float value.
        :param b1: Beta1 parameter in Adam. Float value in range [0-1).
        :param b2: Beta2 parameter in Adam. Float value in range [0-1).
        :param e: epsilon parameter. Float value.
        """
        self.step_size = step_size
        assert type(self.step_size) is FloatType or IntType, "Step size must be an integer or float: {0!r}".format(self.step_size)
        assert 0 < self.step_size, "Step size must be positive: {0!r}".format(self.step_size)
        self.b1 = b1
        assert type(self.b1) is FloatType or IntType, "B1 must be a float or integer: {0!r}".format(self.b1)
        assert 0 <= self.b1 < 1, "B1 must be in range [0, 1): {0!r}".format(self.b1)
        self.b2 = b2
        assert type(self.b2) is FloatType or IntType, "B2 must be a float or integer: {0!r}".format(self.b2)
        assert 0 <= self.b2 < 1, "B2 must be in range [0, 1): {0!r}".format(self.b2)
        self.e = e
        assert type(self.e) is FloatType or IntType, "Epsilon must be an integer or float: {0!r}".format(self.e)
        assert 0 < self.e, "Epsilon must be positive: {0!r}".format(self.e)  
        self.opt_parameters = {'opt': 'Adam', 'alpha':self.step_size, 'b1':self.b1,
                                                                        'b2':self.b2, 'e':self.e}
        logging.info('Optimizer loaded. Type: {0}, step size: {1}, b1: {2},'
            ' b2: {3}, epsilon: {4}'.format(self.opt_parameters['opt'], self.opt_parameters['alpha'],
                self.opt_parameters['b1'], self.opt_parameters['b2'], self.opt_parameters['e']))

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        updates = []
        if gtheta is None:
            assert cost is not None
            gtheta = T.grad(cost, theta)

        fix1 = 1. - self.b1**self.it
        fix2 = 1. - self.b2**self.it
        lr_t = self.step_size * (T.sqrt(fix2) / fix1)

        m = theano.shared(theta.get_value() * 0.)
        v = theano.shared(theta.get_value() * 0.)
        m_t = self.b1 * m + (1-self.b1) * gtheta
        v_t = self.b2 * v + (1-self.b2) * T.sqr(gtheta)
        g_t = m_t / (T.sqrt(v_t) + self.e)
        theta_t = theta - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((theta, theta_t))
        return updates


class SGD(GradientBased):

    def __init__(self, lr_decay_schedule, lr_decay_parameters, momentum_type, momentum=None):
        """
        :param lr_decay_schedule: type learning rate decay schedule used in learning_rate_scheduler.
        :param lr_decay_parameters: used and documented in learning_rate_scheduler.
        :param momentum_type: string either 'classic' or 'nesterov' defining the type of momentum.
        :param momentum: integer or float defining momentum.
        """
        self.lr_update = self.learning_rate_scheduler(lr_decay_schedule, lr_decay_parameters)
        self.momentum_type = momentum_type
        self.momentum_types = ['classic', 'nesterov', 'none']
        assert self.momentum_type in momentum_type, "Momentum type not implemented: {0!r}. Try wiht one of the following: {1}".format(self.momentum_type, self.momentum_types)
        
        assert type(momentum) is FloatType or IntType, "Momentum parameter must be an int or float: {0!r}".format(momentum)
        if self.momentum_type != self.momentum_types[2]:
            assert momentum >=0, "Momentum must be >=0: {0!r}".format(momentum)
        self.momentum_update = lambda epoch: momentum #TODO momentum schedules
        logging.info('SGD optimizer object created')
        self.opt_parameters = {'opt': 'SGD', 'mom':momentum, 'momentum_type':self.momentum_type, 'lr_decay_schedule':lr_decay_schedule, 'lr_decay_parameters':lr_decay_parameters}
        
        logging.info('Optimizer loaded. Type: {0}, momentum_type: {1},'
            ' momentum: {2}, learning rate decay schedule: {3},'
            ' learning rate decay parameters([n(0), r, c]: {4})'.format(self.opt_parameters['opt'],
                self.opt_parameters['momentum_type'],
                self.opt_parameters['mom'], self.opt_parameters['lr_decay_schedule'],
                self.opt_parameters['lr_decay_parameters']))

    def learning_rate_scheduler(self, lr_decay_schedule, lr_decay_parameters):
        """
        Returns a function that gives the learning rate for this epoch taking as input the current epoch
        :param lr_decay_schedule: defines the learning decay schedule.
        Possible schedules:
            constant: n(t) = n(0)
            multiplication: n(t) = n(0)*r^t
            exponential: n(t) = n(0)* 10^(-t/r)
            power: n(t) = n(0)(1+t/t)^-c

        :param lr_decay_parameters: [n(0), r, c]
        """
        #Available learning rate decay schedules
        self.lr_decay_schedules = ['constant', 'multiplication', 'exponential', 'power']
        
        #Checks if asked schedule is implemented
        assert lr_decay_schedule in self.lr_decay_schedules, \
            "Learning rate schedule {0!r} not implemented. Choose one of the following: {1}".format(
                    lr_decay_schedule, self.lr_decay_schedules)

        assert type(lr_decay_parameters) is ListType or TupleType,\
        "Learning rate decay parameters must be a list or a tuple. Provided value: {0!r} and \
        type {1!r}".format(lr_decay_parameters, type(lr_decay_parameters))

        #Initial learning rate value for epoch 0.
        init_lr = lr_decay_parameters[0]

        assert type(init_lr) is FloatType or IntType,\
                        "Initial learning_rate value must be float or int: {0!r}".format(init_lr)

        assert init_lr >= 0, "Initial learning_rate must be >=0: {0!r}".format(init_lr)

        #Constant learning rate
        if lr_decay_schedule == self.lr_decay_schedules[0]:
            return lambda epoch: T.ones(1)*init_lr

        assert len(lr_decay_parameters) >= 2,\
        "If decay not constant at least two parameters [init learning_rate, decay rate] are\
        expected for learning_rate_decay_parameters. Provided values: {0!r}".format(lr_decay_parameters)

        #Parameter r
        decay_rate = lr_decay_parameters[1]

        assert type(decay_rate) is FloatType or IntType,\
                "Decay rate for learning rate value must be float or int: {0!r}".format(decay_rate)

        assert decay_rate >= 0, "Decay rate for learning rate must be >=0: {0!r}".format(decay_rate)

        if lr_decay_schedule == self.lr_decay_schedules[1]:
            return lambda epoch: init_lr*decay_rate**epoch

        if lr_decay_schedule == self.lr_decay_schedules[2]:
            return lambda epoch: init_lr*10**(-epoch*1./decay_rate)

        if lr_decay_schedule == self.lr_decay_schedules[3]:
            assert len(lr_decay_parameters) == 3, "For Power Scheduling 3 parameters are needed \
                                     [init learning rate, r, c]: {0!r}".format(lr_decay_parameters)

            c = lr_decay_parameters[2]
            assert type(c) is FloatType or IntType, \
                "c parameter in Power Schedule for lr must be float or int: {0!r}".format(type(c)) 

            assert c >=0, "c in Power Schedule for lr must be >=0: {0!r}".format(c)
            return lambda epoch: init_lr*(1+epoch*1./r)**-c

    def get_updates(self, theta, cost=None, gtheta=None):
        """Return a list with tuples (variable, update expresion)
        :param cost: theano variable containing the cost to be minimized.
        :param theta: theano sahred variable with the weights to be optimized.
        :param gtheta: theano variable containing the gradients of theta. If None, the gradients are
        computed inside the function. Default: None.
        """
        if gtheta is None:
            assert cost is not None
            #Gradient of weights
            gtheta = T.grad(cost, theta)

        l_r = self.lr_update(self.it)

        mom = self.momentum_update(self.it)  # momentum

        updates = []
        #Previous update needed for momentum. Initalized to 0.
        if self.momentum_type != self.momentum_types[2]:
            v = theano.shared(value=np.zeros(theta.shape.eval(), dtype=theano.config.floatX))
            v_upd = mom * v - l_r * gtheta
            updates.append((v, v_upd))

        if self.momentum_type == self.momentum_types[0]:
            theta_new = theta + v_upd
        elif self.momentum_type == self.momentum_types[1]:
            theta_new = theta + mom * v_upd - l_r * gtheta
        elif self.momentum_type == self.momentum_types[2]:
            theta_new = theta - l_r*gtheta
        else:
            raise NotImplementedError

        updates.append((theta, theta_new))
        self.opt_parameters['lr'] = l_r
        self.opt_parameters['mom'] = mom

        return updates

