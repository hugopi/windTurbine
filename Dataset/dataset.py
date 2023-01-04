import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split


class Dataset:
    """Dataset Class"""

    def __init__(self,name):
        self._dataset = None
        self._name = name
        self._X = None
        self._y = None
        self._X_normalized = None
        self._y_normalized = None
        self._X_scaled = None
        self._y_scaled = None
        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None
        self._X_test = None
        self._y_test = None
        self._minimumBias = None
        self._scaler_x = None
        self._scaler_y = None

    def normalize(self):
        scaler_x = Normalizer().fit(self._X)
        self._X_normalized = scaler_x.transform(self._X)

        scaler_y = Normalizer().fit(self._y)
        self._y_normalized = scaler_y.transform(self._y)

    def scale(self):
        # get only positive value
        self._minimumBias = min(self._y['Power'])
        self._y = self._y + abs(self._minimumBias)
        # scalers
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        # scaling
        self._X_scaled = self._scaler_x.fit_transform(self._X)
        self._y_scaled = self._scaler_y.fit_transform(self._y)

    def preprocess(self, config):
        # split values in train and remaining 80/20
        self._X_train, X_rem, self._y_train, y_rem = train_test_split(self._X_scaled, self._y_scaled,
                                                                      train_size=config['dataset']['train_size'])
        # split X_rem in test and validation 50/50
        self._X_valid, self._X_test, self._y_valid, self._y_test = train_test_split(X_rem, y_rem,
                                                                                    test_size=config['dataset'][
                                                                                        'test_size'])

    def load_data(self, config, remove_zeros=1, frac=0.25):
        """Loads and Preprocess data """
        self._dataset = pd.read_csv(config['dataset']['path'])

        if remove_zeros == 1:
            print("shape before remove zeros : ",self._dataset.shape)
            datasetWO0 = self._dataset[(self._dataset.WindSpeed != 0) & (self._dataset.WindDirection != 0) & (self._dataset.Power != 0)]
            zeros = self._dataset[(self._dataset.WindSpeed == 0) & (self._dataset.WindDirection == 0) & (self._dataset.Power == 0)]
            fracZeros = zeros.sample(frac=frac)
            self._dataset = pd.concat([datasetWO0, fracZeros])
            print("shape after remove zeros : ", self._dataset.shape)

        self._X = self._dataset[['WindSpeed', 'WindDirection']]
        self._y = self._dataset[['Power']]
        #Dataset.normalize(self)
        Dataset.scale(self)
        Dataset.preprocess(self, config)

    def show_hist(self):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.hist(self._X['WindSpeed'], bins=50, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('WindSpeed')
        plt.ylabel('Quantity')
        plt.title('Histogram of windSpeed')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.hist(self._X['WindDirection'], bins=50, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('WindDirection')
        plt.ylabel('Quantity')
        plt.title('Histogram of windDirection')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.hist(self._y['Power'], bins=50, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('Power')
        plt.ylabel('Quantity')
        plt.title('Histogram of Power')
        plt.grid(True)

        plt.suptitle(self._name)
        plt.show()

    @staticmethod
    def lstm_data_transform(x_data, y_data, num_steps=5):
        """ Changes data to the format for LSTM training 
    for sliding window approach """  # Prepare the list for the transformed data
        X, y = list(), list()  # Loop of the entire data set
        for i in range(x_data.shape[0]):
            # compute a new (sliding window) index
            end_ix = i + num_steps  # if index is larger than the size of the dataset, we stop
            if end_ix >= x_data.shape[0]:
                break  # Get a sequence of data for x
            seq_X = x_data[i:end_ix]
            # Get only the last element of the sequency for y
            seq_y = y_data[end_ix]  # Append the list with sequencies
            X.append(seq_X)
            y.append(seq_y)  # Make final arrays
        x_array = np.array(X)
        y_array = np.array(y)
        return x_array, y_array