# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


# From the website https://faroit.github.io/keras-docs/0.1.3/examples/
# The author of the original code (mnist_cnn.py) is Junwei Pan (kemsawill)

from arg import keras_args
import numpy as np
import helper
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from math import sqrt


def keras_ann_predict(model, x_test):
    dim = int(sqrt(len(x_test[0])))
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, dim, dim)
    else:
        x_test = x_test.reshape(x_test.shape[0], dim, dim, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    return model.predict(x_test)


def keras_ann_model(x_train, y_train, epochs=1):
    dim = int(sqrt(len(x_train[0])))
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, dim, dim)
        input_shape = (1, dim, dim)
    else:
        x_train = x_train.reshape(x_train.shape[0], dim, dim, 1)
        input_shape = (dim, dim, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    # param
    num_classes = len(np.unique(y_train))
    batch_size = 1+(len(x_train) // 100)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    #
    # build model
    #
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=1)
    return model


def keras_build_and_predict(data_train, label_train, data_test, epochs):
    model = keras_ann_model(data_train, label_train, epochs)
    model.summary()
    plot_model(model)
    predicted = keras_ann_predict(model, data_test)
    predicted = helper.score_to_class(predicted)
    return predicted


def main():
    args = keras_args()
    rand = np.random.randint(10000000)
    data_train, data_test = helper.pre_processed_data_all(args, rand)
    label_train, label_test = helper.pre_processed_label_all(args, rand)
    res = []
    for i in range(10, 25):
        print('===\n=====Epochs: %d=====\n===' % i)
        resa = []
        for j in range(5):
            resa.append(helper.run_function(keras_build_and_predict,
                                            args.cross_validate,
                                            data_train, label_train,
                                            data_test, label_test,
                                            epochs=i))
        res.append(helper.mean_measures(helper.extract_measures(resa)))
    print(res)
    res = helper.extract_measures(res)
    print(res)
    helper.plot_experiment_server('research', 'epochs', res)


if __name__ == '__main__':
    main()
