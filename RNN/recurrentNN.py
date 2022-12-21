import os
import shutil

import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from Dataset import dataset
from matplotlib import pyplot as plt


class RNN(dataset.Dataset):

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._name = name
        self._history_train = None
        self._history_test = None

    def build(self, config):
        # parameters
        inputLayer = config['model']['input']
        outputLayer = config['model']['output']
        batchSize = config['fit']['batchSize']

        # load data
        RNN.load_data(self, config)

        # create the model
        self._model = tf.keras.Sequential()
        # input_size = 2
        # sequence_length = time step = 6000...
        # batch_input_shape = (batch_size,time step,input_size)
        # units = layer dim = output unit
        # add the input layer

        self._model.add(tf.keras.layers.LSTM(units=128, batch_input_shape=(
            batchSize, self._X_train.shape[0], self._X_train.shape[1]), return_sequences=False, activation='relu'))
        #self._model.add(tf.keras.layers.LSTM(units=128))

        # add the output layer
        self._model.add(
            tf.keras.layers.Dense(outputLayer, kernel_initializer='normal'))

        self._model.summary()

    def run(self, config):
        # parameters
        epochs = config['fit']['epochs']
        loss = config['fit']['loss']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        # optimizer = tf.keras.optimizers.Adam(config['fit']['learningRate'])
        metrics = config['fit']['metrics']

        # compile
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        #x_train = self._X_train.reshape(self._X_train.shape[0], self._X_train.shape[1], 1)
        # train
        self._history_train = self._model.fit(self._X_train, self._y_train,
                                              epochs=epochs, verbose=1)

    def diagnostic(self, config):
        # test
        # Evaluate the model on the test data using `evaluate`
        batchSize = config['fit']['batchSize']
        print("Evaluate on test data")
        results = self._model.evaluate(self._X_test, self._X_test, batch_size=batchSize)
        print("test loss, test MSE:", results)

        # plot loss during training
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(self._history_train.history['loss'], label='train')
        plt.plot(self._history_train.history['val_loss'], label='val')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self._history_train.history['mse'], color='#066b8b', label='train')
        plt.plot(self._history_train.history['val_mse'], color='#b39200', label='val')
        plt.title('model MSE')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend()
        fig.tight_layout()
        plt.show()

        # save trial
        testName = config['model']['testName']
        path = config['model']['saveDirectoryPath']
        target = path + '/' + testName

        if not os.path.exists(target):
            os.makedirs(target)

        # save figure
        plt.savefig(target + '/' + 'diagnostic_plot.png', transparent=True)

        # save config
        src = 'Configs/configs.py'
        shutil.copy(src, target)

        # save history
        epoch = config['fit']['epochs'] - 1
        fichier = open(target + '/' + 'history.txt', "a")

        fichier.write('Training final loss : ')
        fichier.write(f"\n{self._history_train.history['loss'][epoch]}")
        fichier.write('\n')

        fichier.write('\nTraining final MSE : ')
        fichier.write(f"\n{self._history_train.history['mse'][epoch]}")
        fichier.write('\n')

        fichier.write('\nValidation final loss : ')
        fichier.write(f"\n{self._history_train.history['val_loss'][epoch]}")
        fichier.write('\n')

        fichier.write('\nValidation final MSE : ')
        fichier.write(f"\n{self._history_train.history['val_mse'][epoch]}")
        fichier.write('\n')

        fichier.write('\nTest final loss : ')
        fichier.write(f"\n{results[0]}")
        fichier.write('\n')

        fichier.write('\nTest final MSE : ')
        fichier.write(f"\n{results[1]}")
