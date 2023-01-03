from Dataset import dataset
from Configs.configs import CFG
from Model import model
from RNN import recurrentNN

'''myDataset = dataset.Dataset('dataset 1')
myDataset.load_data(CFG, frac=0.25)
myDataset.show_hist()

myDataset2 = dataset.Dataset('dataset 2')
myDataset2.load_data(CFG, remove_zeros=0)
myDataset2.show_hist()

myModel = model.Model('model 1')
myModel.build(CFG)
myModel.run(CFG)
myModel.diagnostic(CFG)
'''
myRNN = recurrentNN.RNN('model 1')
myRNN.build(CFG)
myRNN.run(CFG)
myRNN.diagnostic(CFG)

