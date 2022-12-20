import os
import shutil

import tensorflow as tf
from tensorflow.keras import regularizers
from Dataset import dataset
from matplotlib import pyplot as plt


class Model(dataset.Dataset):

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
        layerDimensions = config['model']['layer_dim']
        layerNames = config['model']['layer_names']
        layerActivation = config['model']['layer_activations']
        layerKernelInitializer = config['model']['layer_kernel_initializer']
        layerKernelRegularizer = config['model']['layer_kernel_regularizer']
        layerActivityRegularizer = config['model']['layer_activity_regularizer']

        # create the model
        self._model = tf.keras.Sequential()

        # add the input layer
        self._model.add(
            tf.keras.layers.Dense(1, input_shape=inputLayer, kernel_initializer='normal', activation='relu',
                                  kernel_regularizer=regularizers.l1(0.01),
                                  bias_regularizer=regularizers.l1(0.01),
                                  activity_regularizer=regularizers.l1(0.01)))
        self._model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.Dropout(0.2)

        for i in range(len(layerDimensions)):
            self._model.add(tf.keras.layers.Dense(layerDimensions[i],
                                                  activation=layerActivation[i],
                                                  name=layerNames[i],
                                                  kernel_regularizer=regularizers.l1(0.01),
                                                  bias_regularizer=regularizers.l1(0.01),
                                                  activity_regularizer=regularizers.l1(0.01)))
            self._model.add(tf.keras.layers.BatchNormalization())
            tf.keras.layers.Dropout(0.2)

        # add the output layer
        self._model.add(
            tf.keras.layers.Dense(outputLayer, kernel_initializer='normal'))

    def run(self, config):
        # parameters
        epochs = config['fit']['epochs']
        loss = config['fit']['loss']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        #optimizer = tf.keras.optimizers.Adam(config['fit']['learningRate'])
        metrics = config['fit']['metrics']
        batchSize = config['fit']['batchSize']

        # load data
        Model.load_data(self, config)

        # compile
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # train
        self._history_train = self._model.fit(self._X_train, self._y_train,
                                              validation_data=(self._X_valid, self._y_valid),
                                              epochs=epochs, batch_size=batchSize, verbose=1)

    def diagnostic(self, config):

        # test
        # Evaluate the model on the test data using `evaluate`
        batchSize = config['fit']['batchSize']
        print("Evaluate on test data")
        results = self._model.evaluate(self._X_test, self._X_test, batch_size=batchSize)
        print("test loss, test acc:", results)

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
