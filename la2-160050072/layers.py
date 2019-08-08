import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		#raise NotImplementedError
		self.data = np.matmul(X , self.weights) + self.biases 
		Y = sigmoid(self.data)
		return Y
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError
		delta = delta * derivative_sigmoid(self.data)
		##  update weights using delta
		new_Delta = delta.dot(np.transpose(self.weights))
		self.weights = self.weights - lr * np.transpose(activation_prev).dot(delta) ### something like this
		for i in range(n):
			self.biases = self.biases - lr * delta[[i]]  
		return new_Delta   
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.numfilters]

		###############################################
		# TASK 1 - YOUR CODE HERE
		#raise NotImplementedError
		self.data = np.zeros((n , self.out_depth , self.out_row , self.out_col))
		# print(self.in_row , self.filter_col)
		# print(X.shape)
		# Z  = X[0][:][0:10][0:10]
		# print(Z.shape)
		# print(Z)
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(0 , self.out_row):
					for y in range(0 , self.out_col):
						ax = x * self.stride 
						ay = y * self.stride
						Z = X[i, : , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)]
						# print(Z.shape , d)
						self.data[i , d , x , y] = sum(sum(sum(Z * self.weights[d]))) +  self.biases[d]														
		# print(X.shape , Y.shape , self.stride)
		return sigmoid(self.data)

		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError
		delta = delta * derivative_sigmoid(self.data)
		##  update weights using delta
		new_Delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		#  what should be the new delta ??

					# increase the weights for each instance of input box by adding input box * delta corresponding 
		# to do, first delta then update
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(0 , self.out_row):
					for y in range(0 , self.out_col):
						ax = x * self.stride 
						ay = y * self.stride
						new_Delta[i , : , ax : (ax + self.filter_row ), ay:(ay + self.filter_col)] +=  delta[i, d, x , y] * self.weights[d, : , : , :] 
						self.weights[d] -=  lr * delta[i , d , x , y] * activation_prev[i , : , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)] 
							# increase the weights for each instance of input box by adding input box * delta corresponding 
		for j in range(self.out_depth):
			self.biases[j] -=   lr * sum(sum(sum(delta[: , j , : , : ])))  
			# should be fine for biases
		return new_Delta  
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.in_channels[2]]

		###############################################
		Y = np.zeros((n , self.out_depth , self.out_row , self.out_col))
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(self.out_row):
					for y in range(self.out_col):
						ax = self.stride * x 
						ay = self.stride * y 
						Z = X[i , d , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)]
						Y[i , d ,x , y] =  np.mean(Z) 													
		return Y
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError
		new_Delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(self.out_row):
					for y in range(self.out_col):
						ax = self.stride * x 
						ay = self.stride * y 
						new_Delta[i , d , ax:ax+self.filter_row , ay:ay+self.filter_col] += delta[i,d,x,y]/(self.filter_row * self.filter_col)
		return new_Delta			 
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))