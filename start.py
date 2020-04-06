import data_utils as du
import matplotlib.pyplot as plt

dataDict = du.get_CIFAR10_data()
xTrain = dataDict["X_train"]
print(xTrain.shape)
print(xTrain.shape[0])