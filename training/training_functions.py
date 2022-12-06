import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras import backend as K
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer
import os
import sys

class CyclicLR(keras.callbacks.Callback):

    def __init__(self, base_lr=1e-5, max_lr=1e-2, step_size=5000):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.scale_fn = lambda x: 1/(2.**(x-1))
        self.scale_mode = 'cycle'
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

def diagonal_nll(y_true, y_pred):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix and ignores constant term.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples (without constant term)
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=diagonal_nll, optimizer='Adam') 
    
    """
    mu = y_pred[:, 0:60]
    twologsigma = y_pred[:, 60:120]
    mse = K.square(y_true-mu)
    weighting = K.exp(-1*twologsigma)
    cost = twologsigma + mse*weighting
    #tf.print("twologsigma: ", tf.math.reduce_max(tf.math.abs(twologsigma)), output_stream=sys.stdout)
    #tf.print("mse: ", tf.math.reduce_max(tf.math.abs(mse)), output_stream=sys.stdout)
    return K.mean(K.sum(cost, axis = 1))


def diagonal_nll2(y_true, y_pred):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix and ignores constant term.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples (without constant term)
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=diagonal_nll2, optimizer='Adam') 
    
    """
    mu = y_pred[:, 0:60]
    twologsigma = y_pred[:, 60:120]
    mse = K.square(y_true-mu)
    weighting = K.exp(-1*twologsigma)
    cost = twologsigma + mse*weighting
    #tf.print("twologsigma: ", tf.math.reduce_max(tf.math.abs(twologsigma)), output_stream=sys.stdout)
    #tf.print("mse: ", tf.math.reduce_max(tf.math.abs(mse)), output_stream=sys.stdout)
    return K.mean(K.sum(cost, axis = 1))

def mse_adjusted(y_true, y_pred):
    mu = y_pred[:, 0:60]
    eval = K.square(mu - y_true)
    return K.mean(eval, axis = 1)

def build_model(hp):
    alpha = hp.Float("leak", min_value = 0, max_value = .4)
    dp_rate = hp.Float("dropout", min_value = 0, max_value = .25)
    batch_norm = hp.Boolean("batch_normalization")
    model = Sequential()
    hiddenUnits = hp.Int("hidden_units", min_value = 128, max_value = 512)
    model.add(Dense(units = hiddenUnits, input_dim=64, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha = alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp.Int("num_layers", min_value = 4, max_value = 11)):
        model.add(Dense(units = hiddenUnits, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha = alpha))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(120, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", ["adam", "RMSprop", "RAdam", "QHAdam"])
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam()
        #optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop()
        #optimizer = keras.optimizers.RMSprop(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam()
        #optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(nu2=1.0, beta1=0.995, beta2=0.999)
        #optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = diagonal_nll2, metrics = [mse_adjusted])
    return model

def set_environment(num_gpus_per_node=4):
    num_gpus_per_node = str(num_gpus_per_node)
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(nodename)
    print(procid)
    stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    output = stream.read()
    oracle = output.split("\n")[0]
    print(oracle)
    if procid==num_gpus_per_node:
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        os.environ["CUDA_VISIBLE_DEVICES"] = procid

    os.environ["KERASTUNER_ORACLE_IP"] = oracle + ".ib.bridges2.psc.edu" # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = "8000"
    print("KERASTUNER_TUNER_ID:    %s"%os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s"%os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s"%os.environ["KERASTUNER_ORACLE_PORT"])
    #print(os.environ)



