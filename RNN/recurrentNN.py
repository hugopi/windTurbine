import os
import shutil

import keras
import tensorflow as tf
from tensorflow.keras import regularizers
from Dataset import dataset
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class RNN(dataset.Dataset):

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._name = name
        self._X_train_lstm = None
        self._y_train_lstm = None
        self._X_valid_lstm = None
        self._y_valid_lstm = None
        self._X_test_lstm = None
        self._y_test_lstm = None
        self._history_train = None
        self._history_test = None

    def build(self, config):
        # parameters
        inputLayer = config['model']['input']
        outputLayer = config['model']['output']
        numberOfSteps = config['model']['numberOfSteps']

        # load data
        RNN.load_data(self, config)
        # reshape for lstm process
        self._X_train_lstm, self._y_train_lstm = RNN.lstm_data_transform(self._X_train, self._y_train,
                                                                         num_steps=numberOfSteps)
        self._X_valid_lstm, self._y_valid_lstm = RNN.lstm_data_transform(self._X_valid, self._y_valid,
                                                                         num_steps=numberOfSteps)
        self._X_test_lstm, self._y_test_lstm = RNN.lstm_data_transform(self._X_test, self._y_test,
                                                                       num_steps=numberOfSteps)

        # create the model
        self._model = tf.keras.Sequential()

        self._model.add(tf.keras.layers.GRU(8, activation='tanh', input_shape=(numberOfSteps, inputLayer[0]),
                                             kernel_regularizer=regularizers.l1(0.01),
                                             bias_regularizer=regularizers.l1(0.01),
                                             activity_regularizer=regularizers.l1(0.01),
                                             return_sequences=True))
        self._model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)

        self._model.add(tf.keras.layers.GRU(16, activation='tanh',
                                             kernel_regularizer=regularizers.l1(0.01),
                                             bias_regularizer=regularizers.l1(0.01),
                                             activity_regularizer=regularizers.l1(0.01),
                                             return_sequences=False))
        self._model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)

        """self._model.add(tf.keras.layers.LSTM(32, activation='tanh',
                                             kernel_regularizer=regularizers.l1(0.01),
                                             bias_regularizer=regularizers.l1(0.01),
                                             activity_regularizer=regularizers.l1(0.01),
                                             return_sequences=False))
        self._model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)"""

        self._model.add(tf.keras.layers.Dense(units=8, activation='relu', kernel_regularizer=regularizers.l1(0.01),
                                              bias_regularizer=regularizers.l1(0.01),
                                              activity_regularizer=regularizers.l1(0.01)))
        self._model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)

        # add the output layer
        self._model.add(
            tf.keras.layers.Dense(outputLayer, activation='linear', kernel_initializer='normal'))

        self._model.summary()

    def run(self, config):
        # parameters
        epochs = config['fit']['epochs']
        loss = config['fit']['loss']
        batchSize = config['fit']['batchSize']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # optimizer = tf.keras.optimizers.Adam(config['fit']['learningRate'])

        metrics = config['fit']['metrics']

        # compile
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # train
        self._history_train = self._model.fit(self._X_train_lstm, self._y_train_lstm,
                                              validation_data=(self._X_valid_lstm, self._y_valid_lstm),
                                              epochs=epochs, batch_size=batchSize, verbose=1)

    def diagnostic(self, config):
        # test
        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        test_predict = self._model.predict(self._X_test_lstm)
        mse = mean_squared_error(test_predict, self._y_test_lstm)
        print("test MSE:", mse)

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
        testName = self._name
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

        fichier.write('\nTest final MSE : ')
        fichier.write(f"\n{mse}")
