import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	c = X.shape[1]
	# print(X.shape)
	X_preprocess = (np.ones((X.shape[0] ,1)))
	for i in range(1, c):
		if isinstance(X[0 , i] , str):
			labels = np.unique(X[: , i])
			features = one_hot_encode(X[: , i] , labels)
			# features = np.zeros(( X.shape[0] , 1))
			X_preprocess = np.c_[ X_preprocess , features]
			continue 
		X_preprocess = np.c_[ X_preprocess , ((X[:,i] - np.mean(X[:,i]))/ (np.std(X[:,i])))]
	# print(X_preprocess)
	return X_preprocess.astype('float64') , Y.astype('float64')
	pass

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''

	# loss = Y - np.matmul(X,W)
	# gradt = np.sum(-2 * X * loss, axis = 0)
	# gradt = np.transpose(loss.T @ X * -2) ; # gradt is a D x 1 matrix 
	gradt = Y + X @ W;
	# grad1 = np.zeros((1 , W.shape[0]))
	# for i in range(W.shape[0]):
	# 	grad1[0 , i] = gradt[i]
	gradient_ridge = gradt + W * 2 * _lambda
	return gradient_ridge 
	pass

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-3):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	N , D = X.shape 
	# y = Y.shape[1]
	W = np.zeros((D , 1))
	T = np.zeros((D , 1))
	X_sm = 2 * X.T @ X 
	Y_sm = -2 * X.T @ Y
	for i in range(max_iter):
		T -= lr * grad_ridge(W , X_sm , Y_sm , _lambda)
		if np.linalg.norm(W - T , 2) < epsilon: 
			break
		W = np.copy(T)
	return W
	pass

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	length = int(X.shape[0]/k)
	error = []
	partitions = []
	data = []
	for i in range(k):
		T = np.copy(X)
		y = np.copy(Y)
		partitions.append(np.delete(T, np.s_[i*length : (i + 1)*length] , 0))
		data.append(np.delete(y, np.s_[i*length : (i + 1)*length] , 0))
	# for i in range(k):
	# 	partitions.append(np.r_[X[0:i*length,:] , X[(i + 1)*length:,:]])
	# 	data.append(np.r_[Y[0:i*length] , Y[(i + 1)*length:]])
	for l in lambdas:
		print("lambda " , l)
		err = 0
		for i in range(k):
			W_train = algo(partitions[i] , data[i] , l )
			err+= (sse(X[i*length:(i+1)*length] , Y[i*length:(i+1)*length] , W_train))
		error.append(err/k)
	# print(error)
	# plot_kfold(lambdas , error)
	return error
	pass

def coord_grad_descent(X, Y, _lambda, max_iter=2000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	d = X.shape[1]
	W = np.ones((d , 1))	
	X2 = 2 * sum(X * X)
	Z = np.transpose(X)
	M = 2 * X.T @ X
	N = 2 * X.T @ Y
	for i in range(max_iter):
		for k in range(d):
			m = X2[k]
			if m == 0:
				W[k,0] = 0 
				continue
			W[k,0] = 0
			# c = 2 * Z[k, :] @ (X @ W - Y)
			c = M[k,:] @ W -N[k]
			if c >= _lambda: 
				W[k , 0 ] =  (1 * _lambda - c )/ m
			elif c <= -1 * _lambda:
				W[k , 0] = (-1 * _lambda - c )/m  
			else:
				W[k , 0] = 0 
			 
	return W 
	pass

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [...] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	lambdas1 = [i for i in range(5,25,1)]  #--- graph is plotted for this.
	lambdas1.append(12.5)
	lambdas1.sort()
	# lambdas1 = [6 , 8, 10, 12.5, 15, 18, 20]
	scores_ridge = k_fold_cross_validation(trainX, trainY, 6, lambdas1, ridge_grad_descent)
	plot_kfold(lambdas1, scores_ridge)
	L_ridge = lambdas1[scores_ridge.index(min(scores_ridge))]
	print("minima_ridge ",  L_ridge )
	W_stud1 = ridge_grad_descent(trainX, trainY, L_ridge)
	sse_ridge = sse(testX, testY, W_stud1)
	print("minimal sse ridge_grad_descent  " , sse_ridge)

	start_l , end_l = 200000 , 600000
	jump = 5000  
	lambdas2 = [i for i in range(start_l , end_l , jump)]  #-- graph is plotted for this.
	# lambdas2 = [390000, 400000 , 415000, 420000 , 425000 , 430000 , 440000 , 450000,  460000 , 480000 , 490000]
	scores_lasso = k_fold_cross_validation(trainX, trainY, 6, lambdas2, coord_grad_descent)
	plot_kfold(lambdas2, scores_lasso)
	L_lasso = lambdas2[(scores_lasso.index(min(scores_lasso)))] 
	print("minima_lasso ",  L_lasso )
	W_stud2 = coord_grad_descent(trainX, trainY, L_lasso)
	sse_Lasso = sse(testX, testY, W_stud2)
	print("minimal sse lasso_grad_descent  " , sse_Lasso)


	print("\ndone\n")