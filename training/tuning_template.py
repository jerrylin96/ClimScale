import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
import keras_tuner as kt
import os

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

set_environment(NUM_GPUS_PER_NODE_HERE)

memory_map = True
batch_size = 10000
num_epochs = 180
shuffle_buffer = 100000

if memory_map:
    train_input = np.load('/dev/shm/train_input.npy', mmap_mode='r')
    train_target = np.load('/dev/shm/train_target.npy', mmap_mode='r')
    val_input = np.load('/dev/shm/val_input.npy', mmap_mode='r')
    val_target = np.load('/dev/shm/val_target.npy', mmap_mode='r')
else:
    train_input = np.load('/dev/shm/train_input.npy')
    train_target = np.load('/dev/shm/train_target.npy')
    val_input = np.load('/dev/shm/val_input.npy')
    val_target = np.load('/dev/shm/val_target.npy')

with tf.device('/CPU:0'):
    train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_target))
    val_ds = tf.data.Dataset.from_tensor_slices((val_input, val_target))

    # Applying transformations to the dataset:
    # Shuffle, batch, and prefetch for the training dataset
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer) # Shuffle the data
    train_ds = train_ds.batch(batch_size, drop_remainder=True)  # Batch the data
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # Prefetch for efficiency

    # Batch and prefetch for the validation dataset
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

def build_model(hp):
    alpha = hp.Float("leak", min_value = 0, max_value = .4)
    dp_rate = hp.Float("dropout", min_value = 0, max_value = .25)
    batch_norm = hp.Boolean("batch_normalization")
    model = Sequential()
    hiddenUnits = hp.Int("hidden_units", min_value = 200, max_value = 480)
    model.add(Dense(units = hiddenUnits, input_dim=175, kernel_initializer='normal'))
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
    model.add(Dense(55, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-3, sampling="log")
    optimizer = hp.Choice("optimizer", ["adam", "RAdam", "QHAdam"])
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(lr_schedule)

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=MAX_TRIALS_HERE,
    executions_per_trial=1,
    overwrite=False,
    directory="tuning_directory/",
    project_name="PROJECT_NAME_HERE",
)

kwargs = {'epochs': num_epochs,
          'verbose': 2,
          'shuffle': True
         }

tuner.search(train_ds, validation_data=val_ds, **kwargs, \
             callbacks=[lr_scheduler, callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True)])


