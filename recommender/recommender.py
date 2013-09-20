#!/usr/bin/python
# -*- coding: utf8 -*-
# This is an adaptation of a python version of exercise 8 for
# Coursera ml-003 machine learning taught by Andrew Ng
# written by Alexander Tronchin-James 2013-05-31

# The algorithm uses regularized least-squares with L-BFGS

import os
import numpy as np
import pylab as pl
from matplotlib import rc; rc('text', usetex=True)
import scipy.io
from scipy.optimize import minimize
import scipy.sparse as sp
import time
import ipdb # not use ipdb.set_trace()

''' Outline of procedures:

 loadData 
    load ratings matrices
    define shapes

 loadLocalData(location?)
    load listing of local availability

 loadUserData(username) 
    load specific user ratings, merge with matrices
    *reduce matrices to union of user ratings and local availability to speedup training

 normalizeRatings
 cofiCostFunc and sparse_mult
 train
    define shapes
    run training
    store result

 predict
    load training result if necessary
    predict for user
    store result
 
 validate
    compare cost on validation and training sets
 tune
    calculate learning curves
    optimize number of features and regularization parameter

'''


## =============== Loading item ratings dataset ================
#  You will start by loading the item ratings dataset to understand the
#  structure of the data.
#  
def loadData(fnam='reviews.pklz2'):
    print 'Loading ratings dataset.\n'

    import gzip,cPickle
    with gzip.open(fnam,'rb') as f: 
        BA=cPickle.load(f)

    user_list = BA['users']
    item_list = BA['beers']

    # Y is a num_items x num_users matrix
    #  cast as float to avoid overflow errors in math
    #  scipy sparse matrices do not throw exceptions on overflow!
    #  http://stackoverflow.com/questions/13402393/scipy-sparse-matrices-inconsistent-sums
    Y = sp.csr_matrix(BA['ratings'].T, dtype=float)

    # R is a num_items x num_users matrix, where R(i,j) = 1 if and only if user j gave a
    # rating to item i
    #  stored R=BA['israted'] has a few extra values? 
    #  maybe there are some zero reviews? Strike these for now.
    R = sp.csr_matrix(sp.csr_matrix(Y,dtype=bool), dtype=float) 

    return {'Y':Y, 'R':R, 'user_list':user_list, 'item_list':item_list}


## ============== User ratings ===============
#  Before we train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed.
#
def loadUserData(user,d,show=False):
    # unpack
    brewer_list = np.array([i[0] for i in d['item_list']])
    beer_list = np.array([i[1] for i in d['item_list']])
    (num_items,num_users) = d['Y'].shape

    # load from disk
    fnam = '../app/userdata/'+user+'.pklz2'
    import gzip,cPickle,json
    with gzip.open(fnam,'rb') as f: 
        udat = cPickle.load(f)
    udat = json.loads(udat)

    # build rows,cols,rats
    rows = [];
    rats = [];
    for u in udat:
        brewer,beer,rat = u
        idx = np.where( (brewer_list==brewer) * (beer_list==beer) )[0]
        if len(idx)>0:
            rows.append(idx[0])
            rats.append(int(rat))
    rows = np.array(rows)
    cols = np.zeros(rows.shape[0])
    rats = np.array(rats)

    # construct my_ratings and myR
    my_ratings = sp.coo_matrix( (rats,(rows,cols)), (num_items,1), dtype=float)

    from copy import copy
    myR=copy(my_ratings)
    myR.data = np.ones(len(myR.data))

    if show:
        item_list = d['item_list']
        print 'User ratings for '+user+':'
        for i in range(len(my_ratings.data)):
            print 'Rated {0} for {1}, {2}'.format(
                my_ratings.data[i],
                brewer_list[my_ratings.row[i]],
                beer_list[my_ratings.row[i]]
                )

    #  Add our own ratings to the data matrix
    d['Y'] = sp.hstack((my_ratings, d['Y']),format='csr')
    d['R'] = sp.hstack((myR, d['R']),format='csr')




## ============ Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.
#
def cofiCostFunc(params, Y,R, shapes, Lambda, debug=False):
    num_users,num_items,num_features,coords = shapes

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

# cython-ized version of sparse_mult: works fast, actually sparse
def sparse_mult(a, b, coords):
    # inspired by handy snippet from 
    # http://stackoverflow.com/questions/13731405/calculate-subset-of-matrix-multiplication
    #a,b are np.ndarrays
    from sparse_mult_c import sparse_mult_c
    rows, cols = coords
    C = np.zeros(rows.shape[0])
    sparse_mult_c(a,b,rows,cols,C)
    return sp.coo_matrix( (C,coords), (a.shape[0],b.shape[1]) )



## ================== Normalize Item Ratings ====================
#
def normalizeRatings(d):
    print '\nNormalizing ratings...'

    # unpack parameters
    Y = d['Y']
    R = d['R']
    m,n = Y.shape
    Rcoo = R.tocoo()
    Rcoords = (Rcoo.row,Rcoo.col)

    nrev = np.array( R.sum(axis=1) ).flatten()
    sumrev = np.array( Y.sum(axis=1) ).flatten()

    Ymean = sumrev / nrev
    d['Ymean'] = Ymean
    d['Ynorm'] = sp.coo_matrix(
        (Y.tocoo().data-Ymean[Rcoords[0]],
         Rcoords
         ), Y.shape
        ).tocsr()



## ================== Training Item Ratings ====================
#
def train(d,
    Lambda = 10, # tunable
    num_features = 10 # tunable
    ):

    if not 'Ynorm' in d.keys(): normalizeRatings(d)

    # unpack parameters
    Y = d['Y']
    Ynorm = d['Ynorm']
    R = d['R']
    m,n = Y.shape
    Rcoo = sp.coo_matrix(R)
    Rcoords = (Rcoo.row,Rcoo.col)

    # build shapes arrays
    (num_items,num_users) = Y.shape
    shapes = (num_users,num_items,num_features,Rcoords)


    print '\nTraining collaborative filtering...\n'

    # Set Initial Parameters (Theta, X), regularization, and costFn
    X = np.random.randn(num_items, num_features)
    Theta = np.random.randn(num_users, num_features)
    initial_parameters = np.hstack((
        np.array(X).flatten('F'), 
        np.array(Theta).flatten('F') 
        ))
    costFn = lambda t: cofiCostFunc( t,Ynorm,R,shapes,Lambda )

    # Train
    result = minimize(costFn, initial_parameters,
        method='L-BFGS-B', jac=True, #callback=cb, 
        options={'disp':True,'maxiter':100}
        )

    print 'Recommender system learning completed.\n'

    # Unfold the returned theta back into X and Theta
    params = result['x']
    d['cost'] = result['fun']
    d['X'] = np.matrix(np.reshape(
        params[:num_items*num_features], 
        (num_items, num_features), order='F'
        ))
    d['Theta'] = np.matrix(np.reshape(
        params[num_items*num_features:],
        (num_users, num_features), order='F'
        ))
    d['Lambda'] = Lambda



## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
def predict(d,show=False,
    predict_user=0 # user number to predict for
    ):

    # unpack parameters
    item_list = d['item_list']
    X = d['X']
    Theta = d['Theta']
    Ymean = d['Ymean']
    Y = d['Y']
    R = d['R']
    Rcoo = sp.coo_matrix(R)
    Rcoords = (Rcoo.row,Rcoo.col)

    # build shapes arrays
    (num_items,num_users) = Y.shape
    num_features = d['Theta'].shape
    shapes = (num_users,num_items,num_features,Rcoords)

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

    if show:
        print '\nTop recommendations for you:'
        for i in range(10):
            j = ix[i]
            print 'Predicting rating {0} for item {1}'.format(
                my_predictions[j],
                item_list[j]
                )

    d['predictions'] = my_predictions


def run():
    d = loadData()
    loadUserData(user,d,show=True)
    normalizeRatings(d)
    train(d)
    predict(d,show=True)

