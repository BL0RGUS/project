#!/usr/bin/env python3
# coding: utf-8

# # MNIST Classification
# ## A Convolutional Neural Network Demo

import os

##### For reproducibility
seed_value= 1
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K

from tensorflow.keras import utils, losses
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import h5py

# define square activation
class SquareActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.square(inputs)

batch_size = 500
epochs = 10
print("Misc. initializations")


# ### Data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train /= 255
x_test /= 255
print('data ready')

print('shape: ',x_train.shape)


# Create validation data
testSize=16
x_val = x_test[:-testSize]
x_test = x_test[-testSize:]
y_val = y_test[:-testSize]
y_test = y_test[-testSize:]
print('Validation and test data ready')

# Convert class vector to binary class matrices
num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
y_val = utils.to_categorical(y_val, num_classes)

input_shape = x_train[0].shape
print(f'input shape: {input_shape}')


# ### Save dataset
def save_data_set(x, y, data_type, s=''):
    fname=f'x_{data_type}{s}.h5'
    print("Saving x_{} of shape {} in {}".format(data_type, x.shape,fname))
    xf = h5py.File(fname, 'w')
    xf.create_dataset('x_{}'.format(data_type), data=x)
    xf.close()

    yf = h5py.File(f'y_{data_type}{s}.h5', 'w')
    yf.create_dataset(f'y_{data_type}', data=y)
    yf.close()

save_data_set(x_test, y_test, data_type='test')
# save_data_set(x_train, y_train, data_type='train')
# save_data_set(x_val, y_val, data_type='val')


model = Sequential()
model.add(Conv2D(filters=5, kernel_size=5, strides=2, padding='valid',
                 input_shape=input_shape))
model.add(Flatten())
model.add(SquareActivation())
model.add(Dense(100))
model.add(SquareActivation())
model.add(Dense(num_classes))

model.summary()


# ### Train
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

model.compile(loss=sum_squared_error,
                  optimizer='Adam',
                  metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_val, y_val),
              shuffle=True,
              )
score = model.evaluate(x_test, y_test, verbose=0)

print(f'Test loss: { score[0]:.3f}')
print(f'Test accuracy: {score[1] * 100:.3f}%')


# ### Confusion Matrix
from sklearn import metrics

y_pred_vals = model.predict(x_test)
y_pred = np.argmax(y_pred_vals, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)


# ### Serialize model and weights
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('model.h5')
print("Saved model to disk")
