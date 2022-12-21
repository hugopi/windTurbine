"""Model config in json format"""

CFG = {
    "dataset": {
        "path": "D:/PycharmProjects/IPSA/Aero5/windTurbine/Ressources/Turbine_Data_project.csv",
        "train_size": 0.8,
        "test_size": 0.5
    },
    "fit": {
        "batchSize": 16384,
        "epochs": 200,
        "loss": 'mean_squared_logarithmic_error',
        "learningRate": 0.001,
        "metrics": ["mse"]
    },
    "model": {
        "input": [2],
        "output": 1,
        "layer_dim": [10, 128, 256, 128, 64,8],
        "layer_names": ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5','layer_6'],
        "layer_activations": ['relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
        "layer_kernel_initializer": ['normal', 'normal', 'normal', 'normal', 'normal', 'normal'],
        "layer_kernel_regularizer": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "layer_activity_regularizer": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "saveDirectoryPath": 'savings',
        "testName": 'first_RNN'
    }
}