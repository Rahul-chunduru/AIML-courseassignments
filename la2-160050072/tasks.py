import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	# raise NotImplementedError
	nn1 = nn.NeuralNetwork(2, 0.1, 20, 50)
	nn1.addLayer(FullyConnectedLayer(2 , 4)) # seed is 61 for 4 nodes
	nn1.addLayer(FullyConnectedLayer(4 , 2))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	# raise NotImplementedError
	###############################################
	nn1 = nn.NeuralNetwork(2, 0.1, 20, 30)
	nn1.addLayer(FullyConnectedLayer(2 , 2)) # seed is 62
	nn1.addLayer(FullyConnectedLayer(2 , 2))
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	# raise NotImplementedError	
	nn1 = nn.NeuralNetwork(10, 0.1, 30, 15)
	nn1.addLayer(FullyConnectedLayer(784 , 20))
	nn1.addLayer(FullyConnectedLayer(20 , 10))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	
	XTrain = XTrain[0:5000,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	# XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	# YTest = YTest[0:1000,:]
	YTrain = YTrain[0:5000,:]
	

	lr = 0.2
	batchSize = 10
	epochs = 20
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	# raise NotImplementedError	1
	###################################################
	
	# Reserve
	# nn1 = nn.NeuralNetwork(10, 0.2, 10, 30)
	# nn1.addLayer(ConvolutionLayer([3,32,32], [10,10], 4, 2))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(576,50))
	# # nn1.addLayer(FullyConnectedLayer(100,50))
	# nn1.addLayer(FullyConnectedLayer(50,10))

	print("Used parameters: ")
	print("learning rate" , lr)
	print("batchSize" , batchSize)
	print("epochs" , epochs)
	nn1 = nn.NeuralNetwork(10, 0.2, 20, 30)
	nn1.addLayer(ConvolutionLayer([3,32,32], [10,10], 4, 2))
	nn1.addLayer(AvgPoolingLayer([4,12,12], [4,4], 4))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(36,18))
	nn1.addLayer(FullyConnectedLayer(18,10))



	return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION
	# nn1 = nn.NeuralNetwork(10, lr, batchSize, epochs)
	# nn1.addLayer(ConvolutionLayer([3,32,32], [10,10], 4, 2))
	# # nn1.addLayer(AvgPoolingLayer([8,14,14], [2,2], 2))
	# nn1.addLayer(FlattenLayer())
	# nn1.addLayer(FullyConnectedLayer(576,300))
	# nn1.addLayer(FullyConnectedLayer(300,100))	
	# nn1.addLayer(FullyConnectedLayer(100,10))
	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=False, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
