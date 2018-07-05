#!/usr/bin/python
# Matrix Factorization 
# simple example to demonstrate matrix factorization

import numpy as n

# Accept array and dim of number of  attributes 
def factorize(arr, dim):
	print arr
	print dim
	alpha = 0.002
	steps = 5000

	# get the shape of input array
	(rows, columns) = arr.shape	
	print rows, columns

	# arr = left x right.transpose
	left = n.random.rand(rows, dim)
	right = n.random.rand(columns, dim)
	right_tran = right.T
	
	# Gradient decent
	# converge in --steps-- interations
	for step in range(steps):
		for i in range(rows):
			for j in range(columns):
				# consider only the non zero values in the sparse matrix
				if arr[i][j] > 0:
					eij = arr[i][j] - n.dot(left[i,:], right_tran[:,j])
					for d in range(dim):
						left[i][d] = left[i][d] + alpha * (2 * eij * right_tran[d][j])
						right_tran[d][j] = right_tran[d][j] + alpha * (2 * eij * left[i][d])
		
		# estimate the error
		eArr = 0
		for i in range(rows):
			for j in range(columns):
				eArr = eArr + (arr[i][j] - n.dot(left[i,:],right_tran[:,j])) ** 2

		# stop when error is small enough
		if eArr < 0.001:
			print "steps executed - ", step
			break

	return left, right

# Accept array and dim of number of  attributes an regularize
def factorize_with_regularization(arr, dim):
	print arr
	print dim
	alpha = 0.002
	steps = 5000
	beta = 0.02

	# get the shape of input array
	(rows, columns) = arr.shape	
	print rows, columns

	# arr = left x right.transpose
	left = n.random.rand(rows, dim)
	right = n.random.rand(columns, dim)
	right_tran = right.T
	
	# Gradient decent
	# converge in --steps-- interations
	for step in range(steps):
		for i in range(rows):
			for j in range(columns):
				# consider only the non zero values in the sparse matrix
				if arr[i][j] > 0:
					eij = arr[i][j] - n.dot(left[i,:], right_tran[:,j])
					for d in range(dim):
						left[i][d] = left[i][d] + alpha * (2 * eij * right_tran[d][j]) - beta * left[i][d]
						right_tran[d][j] = right_tran[d][j] + alpha * (2 * eij * left[i][d]) - beta * right_tran[d][j]
		
		# estimate the error
		eArr = 0
		for i in range(rows):
			for j in range(columns):
				eArr = eArr + (arr[i][j] - n.dot(left[i,:],right_tran[:,j])) ** 2
				for d in range(dim):
					e1 = (left[i][d]**2 + right_tran[d][j]**2)
					eArr = eArr + (beta/2) * e1

		# stop when error is small enough
		if eArr < 0.001:
			print "steps executed - ", step
			break

	return left, right

# Implement SVD
def matrix_svd(inp, dim):
	u, s, v = n.linalg.svd(inp, full_matrices=False)
	#u, s, v = n.linalg.svd(n.array(inp), full_matrices=True)
	u = u[:,:dim] 
	s = s[:dim] 
	v = v[:dim,:]	

	return u, s, v
	
# Calculate RMSE
def rmse(inp, out):
	err = 0
	rows, columns = n.shape(inp)
	for i in range(rows):
		for j in range(columns):
			if (inp[i][j] > 0):
				err = err + (inp[i][j] - out[i][j]) ** 2

	return err

if __name__ == "__main__":
	print "start"
	inp = [ [5,3,0,0,4,0,0,0,0],
		[0,0,2,3,0,1,5,0,2],
		[4,0,0,0,0,3,0,0,2],
		[1,0,0,4,1,0,0,0,0],
		[0,1,5,0,2,1,4,0,1],
		[0,0,3,0,0,0,3,0,0],
		[0,0,0,0,0,2,1,0,0],
		[0,0,0,0,0,0,0,0,2],
		[0,0,0,0,0,2,0,0,0],
		[0,2,0,0,0,0,1,0,1],
		[2,0,0,5,0,0,0,5,0],
		[1,0,0,0,0,2,0,1,2],
		[0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,1,1,1],
		[0,0,0,0,0,2,1,0,0],
		[0,1,0,0,0,0,0,0,1],
		[4,0,1,0,0,3,0,0,0]]

	users = 50
	movies = 20
	inp = n.round(n.random.rand(users,movies) * 10) % 5
	i = 0
	while i < 800:
		x = n.random.rand(1,1)*10 
		y = n.random.rand(1,1)*10 
		inp[int(x[0,0]), int(y[0,0])] = 0
		i = i + 1

	print "non zero elemets = ", n.count_nonzero(inp)

	latent_attrib = 4
	(l,r) = factorize(n.array(inp), latent_attrib)
	recco1 = n.dot(l, r.T)
	print "********************MF**********************"
	print "User Characteristics"
	print n.round(l)
	print "Item Characteristics"
	print n.round(r.T)
	print "Predicted recommendations"
	print n.round(recco1)
	print "ERROR = ", rmse(inp, n.array(recco1))
	print "********************MF**********************"
	#(l,r) = factorize_with_regularization(n.array(inp), latent_attrib)
	#print n.dot(l, r.T)

	# use builtin SVD 
	d = 2
	u, s, v = matrix_svd(inp, d)
	# reconstruct - u x diag(s) x v: note - v is the transpose of V from SVD
	recco2 = n.dot(n.dot(u, n.diag(s)), v)
	print "********************SVD**********************"
	print "User Characteristics"
	print n.round(u)
	print "Diagonal"
	print n.round(s)
	print "Item Characteristics"
	print n.round(v)
	print "Predicted recommendations"
	print n.round(recco2)
	print "ERROR = ", rmse(inp, n.array(recco1))
	print "********************SVD**********************"

