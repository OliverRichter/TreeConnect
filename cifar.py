from __future__ import print_function
from keras.layers import Dense, Activation, Flatten, Reshape, Permute, LocallyConnected1D
from keras.layers import Convolution2D
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import callbacks
from keras.engine import Layer
from keras.datasets import cifar10
import numpy as np

batch_size = 64
nb_classes = 10
nb_epoch = 50
nb_runs = 5
learning_rate = 0.001
architecture = 'TreeConnect'  # can be one of 'Shallow', 'Full', 'TreeConnect', 'Bottleneck', 'Small' or 'RandomSparse'

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


class RandomSparseLayer(Layer):
    def __init__(self, output_dim, percentage_masked, **kwargs):
        self.output_dim = output_dim
        self.fraction_masked = percentage_masked / 100.0
        super(RandomSparseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = (input_shape[1], self.output_dim)
        self.weight_mask = np.where(np.random.rand(*weight_shape) < self.fraction_masked, np.zeros(weight_shape),
                                    np.ones(weight_shape))
        self.weight = self.add_weight(name='weight', shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        super(RandomSparseLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return Activation('relu')(K.dot(inputs, self.weight * self.weight_mask) + self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


# Pre-processing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

for run_id in range(nb_runs):
    model = Sequential()
    if architecture == 'Small':
        model.add(Convolution2D(8, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
        model.add(Convolution2D(8, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(16, (4, 4), padding='same', strides=2, activation='relu'))
        model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(32, (4, 4), padding='same', strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(552))
        model.add(Dense(256))
    else:
        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(128, (4, 4), padding='same', strides=2, activation='relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(256, (4, 4), padding='same', strides=2, activation='relu'))
    if architecture == 'Shallow':
        model.add(Flatten())
        model.add(Dense(256))
    elif architecture == 'Full':
        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Dense(256))
    elif architecture == 'TreeConnect':
        model.add(Reshape((128, 128)))
        model.add(LocallyConnected1D(16, 1, activation='relu'))
        model.add(Permute((2, 1)))
        model.add(LocallyConnected1D(16, 1, activation='relu'))
        model.add(Flatten())
    elif architecture == 'Bottleneck':
        model.add(Flatten())
        model.add(Dense(18))
        model.add(Dense(167))
    elif architecture == 'RandomSparse':
        model.add(Flatten())
        model.add(RandomSparseLayer(2048, percentage_masked=99.22))
        model.add(RandomSparseLayer(256, percentage_masked=93.75))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    tbCallback = callbacks.TensorBoard(log_dir='./cifar/' + architecture + '_run_' + str(run_id),
                                       histogram_freq=0, write_graph=False)
    model.summary()
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[tbCallback])
    K.clear_session()
