#!/usr/bin/python
# -*- coding: utf8 -*-
# This is an adaptation of a python version of exercise 8 for
# Coursera ml-003 machine learning taught by Andrew Ng
# written by Alexander Tronchin-James 2013-05-31

# The algorithm uses a regularized least-squares cost function and L-BFGS for optimization

import os
import numpy as np
import pylab as pl
from matplotlib import rc; rc('text', usetex=True)
import scipy.io
from scipy.optimize import minimize
import scipy.sparse as sp
import time
import ipdb # not use ipdb.set_trace()


## Machine Learning Online Class
#  Exercise 8 | Collaborative Filtering / recommender systems
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#

## Initialization
os.system('clear') # clear the screen
pl.close('all') # close any open plots
pl.ion() # turn on plotting interactive mode, otherwise plots pause computation


## =============== Part 1: Loading item ratings dataset ================
#  You will start by loading the item ratings dataset to understand the
#  structure of the data.
#  
print 'Loading ratings dataset.\n'

#  Load data
import gzip,cPickle
def loadData(fnam):
    with gzip.open(fnam,'rb') as f: return cPickle.load(f)

BA = loadData('reviews.pklz2')

# cast as floats to avoid overflow errors in math
# scipy sparse matrices do not throw exceptions on overflow!
# http://stackoverflow.com/questions/13402393/scipy-sparse-matrices-inconsistent-sums
Y = sp.csr_matrix(BA['ratings'].T, dtype='float')
# stored 'israted' has some extra values? 
# Maybe there are some zero reviews? Strike these for now.
R = sp.csr_matrix(sp.csr_matrix(Y,dtype=bool), dtype=float) 

# build shapes arrays
Rcoo = sp.coo_matrix(R)
Rcoords = (Rcoo.row,Rcoo.col)
(num_items,num_users) = Y.shape
num_features = 10 # adjustable
shapes = (num_users,num_items,num_features)

shapes = (shapes,Rcoords)

#  Y is a num_items x num_users matrix, containing ratings (1-5) of 1682 items on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to item i

user_list = BA['users']
item_list = BA['beers']

#print 'Program paused. Press enter to continue.'; raw_input()


## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

# Define cost function
def cofiCostFunc(params, Y,R, shapes, Lambda, debug=False):
    num_users, num_items, num_features = shapes[0]
    coords = shapes[1]

    # reshape params into X and Theta
    X = np.reshape(
        params[:num_items*num_features], 
        (num_items, num_features), order='F'
        )
    Theta = np.reshape(
        params[num_items*num_features:],
        (num_users, num_features), order='F'
        )
    Lambda=float(Lambda) # ensure Lambda isn't an int!
    
    # do sparse multiply for the below, see last answer
    # http://stackoverflow.com/questions/13731405/calculate-subset-of-matrix-multiplication
    #XThmY = X*Theta.T - Y
    #XThmYR= R.multiply(XThmY) # multiply method of sparse matrices is element-wise
    # already should be encoded by the sparsity pattern enforced above
    XThmYR = sparse_mult(X,Theta.T,coords) - Y

    J = 1/2. *(XThmYR.data**2).sum()
    J = J + Lambda/2.*( (np.array(Theta)**2).sum() + (np.array(X)**2).sum() )

    X_grad = XThmYR*Theta + Lambda*X
    Theta_grad = XThmYR.T*X + Lambda*Theta

    if debug:
        print 'X = \n{0}'.format(X)
        print 'Theta = \n{0}'.format(Theta)
        print 'X_grad = \n{0}'.format(X_grad)
        print 'Theta_grad = \n{0}'.format(Theta_grad)

    # Unroll gradients
    grad = np.hstack((
        np.array(X_grad).flatten('F'),
        np.array(Theta_grad).flatten('F')
        ))
    J = float(J)
    return (J,grad)

# cython-ized version, works fast!
def sparse_mult(a, b, coords):
    # inspired by handy snippet from 
    # http://stackoverflow.com/questions/13731405/calculate-subset-of-matrix-multiplication
    #a,b are np.ndarrays
    from sparse_mult_c import sparse_mult_c
    rows, cols = coords
    C = np.zeros(rows.shape[0])
    sparse_mult_c(a,b,rows,cols,C)
    return sp.coo_matrix( (C,coords), (a.shape[0],b.shape[1]) )


''' skip new user data for now, try to get large set working first
## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  items in our dataset!
#

#  Initialize my ratings
my_ratings = sp.lil_matrix((num_items,1))

# Check the file item_idx.txt for id of each item in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0,0] = 4

# Or suppose did not enjoy Silence of the Lambs [1991,0], you can set
my_ratings[97,0] = 2

# We have selected a few items we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6,0] = 3
my_ratings[11,0]= 5
my_ratings[53,0] = 4
my_ratings[63,0]= 5
my_ratings[65,0]= 3
my_ratings[68,0] = 5
my_ratings[182,0] = 4
my_ratings[225,0] = 5
my_ratings[354,0]= 5

from copy import copy
my_ratings=my_ratings.tocsr()
myR=copy(my_ratings)
myR.data = np.ones(len(myR.data))

def loadItemList():
    with open('item_ids.txt','r') as f:
        item_list = f.readlines()
    return np.array([' '.join(l.split()[1:]) for l in item_list])
item_list = loadItemList()

print 'New user ratings:'
def print_my_ratings():
    for i in range(len(my_ratings.data)):
        if my_ratings.data[i] > 0:
            print 'Rated {0} for {1}'.format(my_ratings.data[i],item_list[my_ratings.tocoo().row[i]])
print_my_ratings()

#  Add our own ratings to the data matrix
Y = sp.hstack((my_ratings, Y),format='csr')
R = sp.hstack((myR, R),format='csr')

#  Re-calculate updated shapes
num_items,num_users = Y.shape
num_features = 10 # adjustable
shapes = (num_users,num_items,num_features)

Rcoo = sp.coo_matrix(R)
Rcoords = (Rcoo.row,Rcoo.col)
shapes = (shapes,Rcoords)

#print 'Program paused. Press enter to continue.'; raw_input()
'''


## ================== Part 7: Learning Item Ratings ====================
#  Now, you will train the collaborative filtering model on a item rating 
#  dataset of 1682 items and 943 users
#

#  Normalize Ratings
def normalizeRatings(Y,R):
    m,n = Y.shape

    nrev = np.array( R.sum(axis=1) ).flatten()
    sumrev = np.array( Y.sum(axis=1) ).flatten() 
    Ymean = sumrev / nrev
    Ynorm = sp.coo_matrix((Y.tocoo().data-Ymean[Rcoords[0]],Rcoords),Y.shape).tocsr()
    return (Ynorm,Ymean)

print '\nNormalizing ratings...'
Ynorm,Ymean = normalizeRatings(Y, R);


# Set Initial Parameters (Theta, X), regularization, and costFn
X = np.random.randn(num_items, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.hstack((
    np.array(X).flatten('F'), 
    np.array(Theta).flatten('F') 
    ))
Lambda = 10 # adjustable
costFn = lambda t: cofiCostFunc( t,Ynorm,R,shapes,Lambda )

# Train
print '\nTraining collaborative filtering...\n'
result = minimize(costFn, initial_parameters,
                  method='L-BFGS-B', jac=True, #callback=cb, 
                  options={'disp':True,'maxiter':100}
                 ) 
cost,params = (result['fun'],result['x'])

# Unfold the returned theta back into X and Theta
X = np.matrix(np.reshape(
    params[:num_items*num_features], 
    (num_items, num_features), order='F'
    ))
Theta = np.matrix(np.reshape(
    params[num_items*num_features:],
    (num_users, num_features), order='F'
    ))

print 'Recommender system learning completed.\n'

#print 'Program paused. Press enter to continue.'; raw_input()


## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

predict_user = 0 # user number to predict for
pcoords = (
           np.array( range(num_items) ),
           predict_user + np.zeros(num_items,dtype='int')
          )
p = sparse_mult( X,Theta.T, pcoords ).tocsr()
my_predictions = np.array(p[:,predict_user].todense()).flatten() + Ymean

ix = [i[0] for i in sorted(
    enumerate(my_predictions), 
    key=lambda x:x[1], 
    reverse=True
    )]

print '\nTop recommendations for you:'
for i in range(10):
    j = ix[i]
    print 'Predicting rating {0} for item {1}'.format(
        my_predictions[j],
        item_list[j]
        )

