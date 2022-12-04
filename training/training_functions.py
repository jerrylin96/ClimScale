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
    #the weighting has been modified for stable training.
    weighting = tf.math.expm1(-1*twologsigma) + 1
    cost = twologsigma + mse*weighting
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
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
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



