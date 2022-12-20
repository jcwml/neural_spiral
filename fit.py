# github.com/jcwml
import sys
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from random import seed
from time import time_ns
from sys import exit
from os.path import isfile
from os import mkdir
from os.path import isdir

# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")
# print(device_lib.list_local_devices())
# print(tf.config.list_physical_devices())
# exit();

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
//os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# hyperparameters
seed(74035)
model_name = 'keras_model'
optimiser = 'nesterov'
inputsize = 1
outputsize = 2
epoches = 6600
activator = 'tanh'
layers = 16
layer_units = 32
batches = 1
samples = 33

# load options
argc = len(sys.argv)
if argc >= 2:
    layers = int(sys.argv[1])
    print("layers:", layers)
if argc >= 3:
    layer_units = int(sys.argv[2])
    print("layer_units:", layer_units)
if argc >= 4:
    batches = int(sys.argv[3])
    print("batches:", batches)
if argc >= 5:
    activator = sys.argv[4]
    print("activator:", activator)
if argc >= 6:
    optimiser = sys.argv[5]
    print("optimiser:", optimiser)
if argc >= 7 and sys.argv[6] == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("CPU_ONLY: 1")
if argc >= 8:
    samples = int(sys.argv[7])
    print("samples:", samples)
if argc >= 9:
    epoches = int(sys.argv[8])
    print("epoches:", epoches)

# make sure save dir exists
if not isdir('models'): mkdir('models')
model_name = 'models/' + activator + '_' + optimiser + '_' + str(layers) + '_' + str(layer_units) + '_' + str(batches) + '_' + str(samples) + '_' + str(epoches)

##########################################
#   CREATE DATASET
##########################################
print("\n--Creating Dataset")
st = time_ns()

train_x = np.empty([samples, 1], float)
train_y = np.empty([samples, 2], float)

sp = 1.0 / float(samples)
for i in range(samples):
    m = sp * float(i)
    train_x[i] = m
    train_y[i] = [np.sin(m*33.0)*333.0*m, np.cos(m*33.0)*333.0*m]

shuffle_in_unison(train_x, train_y)
# train_x = np.reshape(train_x, [samples, inputsize])
# train_y = np.reshape(train_y, [samples, outputsize])

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################
print("\n--Training Model")

# construct neural network
model = Sequential()

model.add(Dense(layer_units, activation=activator, input_dim=inputsize))

for x in range(layers):
    # model.add(Dropout(.3))
    model.add(Dense(layer_units, activation=activator))

# model.add(Dropout(.3))
model.add(Dense(outputsize))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'sgd':
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*samples, decay_rate=0.1)
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*samples, decay_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)

model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

# train network
history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches)
model_name = model_name + "_" + "[{:.2f}]".format(history.history['accuracy'][-1])
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################
print("\n--Exporting Model")
st = time_ns()

# save prediction model
predict_x = np.empty([8192, 1], float)
sp = 1.0 / 8192.0
for i in range(8192):
    predict_x[i] = sp * float(i)

f = open(model_name + "_pd.csv", "w")
if f:

    p = model.predict(predict_x)
    for i in range(8192):
        f.write(str(sp*float(i)) + "," + str(p[i][0]) + "," + str(p[i][1]) + "\n")

    # sp = 1.0 / 8192.0
    # for i in range(8192):
    #     pi = sp*float(i)
    #     input = np.reshape(pi, [-1, 1])
    #     p = model.predict(input)
    #     r = p.flatten()
    #     f.write(str(pi) + "," + str(r[0]) + "," + str(r[1]) + "\n")

    f.close()

# save keras model
# model.save(model_name)

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds\n")
