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
#import ipdb # for ipdb.set_trace()

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
dat_dir = ''
def loadData(fnam='reviews.pklz2'):
    print 'Loading ratings dataset.\n'

    # search for data files, changes depending on invoking from 
    #  import app.recommender as rec (i.e. dat_dir='')
    #    vs.
    #  import recommender as rec (i.e. dat_dir='app')
    import os
    print 'current directory: '+os.getcwd()
    global dat_dir
    if os.path.exists('app'):
        dat_dir='app/'
        print ' getting data from: '+dat_dir

    import gzip,cPickle
    with gzip.open(dat_dir+fnam,'rb') as f: 
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
    fnam = 'userdata/'+user+'.pklz2'
    import gzip,cPickle,json
    with gzip.open(dat_dir+fnam,'rb') as f: 
        udat = cPickle.load(f)
    udat = json.loads(udat['webdat'])

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
    d['user_list'] = np.hstack((user,d['user_list']))


    # filter to local and user reviewed items
    with gzip.open(dat_dir+'match_BA_RB.pklz2','rb') as f: 
        RB_matches=cPickle.load(f)

    rows_filt = np.array(sorted(set( [i[0] for i in RB_matches] + list(rows) ))) # local beers + user reviewed beers
    Y = d['Y'][rows_filt,:]
    R = d['R'][rows_filt,:]

    user_nrev = np.array(R.sum(axis=0)).flatten()
    cols_filt = np.where( user_nrev>=0 )[0] # filter columns (users) with no reviews
    Y = Y[:,cols_filt]
    R = R[:,cols_filt]

    # also filter users with more than 100 reviews

    # store filtered results
    d['RB_matches']=RB_matches
    d['Y']=Y
    d['R']=R
    d['user_list']=d['user_list'][cols_filt]
    d['item_list']=d['item_list'][rows_filt]
    d['cols_filt']=cols_filt
    d['rows_filt']=rows_filt



## ============ Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.
#
from sparse_mult import sparse_mult

def cofiCostFunc(params, Y,R, shapes, Lambda, debug=False, validate=False):
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

    Jerr = 1/2. *(XThmYR.data**2).sum()

    if validate: # return only validation error
        Jval = np.sqrt(Jerr*2./XThmYR.nnz) # RMS error
        return Jval

    else:
        # regularize
        J = Jerr + Lambda/2.*( (np.array(Theta)**2).sum() + (np.array(X)**2).sum() )

        # gradients
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

    Ymean = sumrev / nrev # average review for each beer
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


# Calculate standard error
def stdError(d):
    # unpack parameters
    Lambda = d['Lambda']
    Y = d['Y']
    Ynorm = d['Ynorm']
    R = d['R']
    m,n = Y.shape
    Rcoo = sp.coo_matrix(R)
    Rcoords = (Rcoo.row,Rcoo.col)
    X = d['X']
    Theta = d['Theta']

    # build shapes arrays
    (num_items,num_users) = Y.shape
    num_features = d['Theta'].shape[1]
    shapes = (num_users,num_items,num_features,Rcoords)

    pars = np.hstack((
        np.array(X).flatten('F'), 
        np.array(Theta).flatten('F') 
        ))

    err = cofiCostFunc( pars,Ynorm,R,shapes,Lambda,validate=True )
    return err


## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#
def predict(d,show=False,
    predict_user=0 # user number to predict for
    ):

    # unpack parameters
    item_list = d['item_list']
    R = d['R']
    Y = d['Y']
    X = d['X']
    Theta = d['Theta']
    Ymean = d['Ymean']
    (num_items,num_users) = Y.shape

    # calculate predictions
    pcoords = ( # need to specify dtype='int32' for sparse_mult
               np.array( range(num_items),dtype='int32'),
               predict_user + np.zeros(num_items,dtype='int32')
              )
    p = sparse_mult( X,Theta.T, pcoords ).tocsr()
    my_predictions = np.array(p[:,predict_user].todense()).flatten() + Ymean

    # filter out reviewed beers
    new_recs = np.where( np.array(R[:,0].todense()).flatten()==0 )[0] # e.g. beers not reviewed
    my_predictions = np.array(list(enumerate(my_predictions)))[ new_recs ]
    d['rows_filt_new'] = d['rows_filt'][new_recs]

    ix = [i[0] for i in sorted(
        enumerate(my_predictions),
        key=lambda x:x[1][1], 
        reverse=True
        )]

    if show:
        print 'Top recommendations for you:'
        for i in range(10):
            j = ix[i]
            print 'Predicting rating {0} for item {1}'.format(
                my_predictions[j],
                item_list[j]
                )

    d['predictions'] = my_predictions


def findBestLocations(d):
    import gzip,cPickle
    # this loads {'area_url':url,'locations':locations,'beers':beers}

    with gzip.open(dat_dir+'scrape_ratebeer.pklz','rb') as f: RB=cPickle.load(f)
    with gzip.open(dat_dir+'match_BA_RB.pklz2','rb') as f: matches=cPickle.load(f) 
    # set(matches) is smaller due to bad matches
    
    # match BA predictions to the RB beers
    user_beers = RB['beers']
    for i in range(len(matches)):
        idx = np.where( matches[i][0] == d['rows_filt_new'] )[0]
        #print i,idx,matches[i][0]
        if len(idx)>0:
            user_beers[i]['prediction'] = d['predictions'][idx[0]][1]

    # match each beer in each location to a BA prediction
    urls = np.array([u['url'] for u in user_beers])
    user_loc = RB['locations']
    for l in user_loc:
        # assign beer predictions
        for b in l['beers']:
            idx = np.where( urls==b['item']['url'] )[0]
            if len(idx)>0 and ('prediction' in user_beers[idx]):
                b['item']['prediction'] = user_beers[idx]['prediction']
            else: # define prediction of zero for already reviewed beers or beers without match
                b['item']['prediction'] = 0.

        # sort beers descending by prediction
        l['beers'].sort( key=lambda x:x['item']['prediction'],reverse=True )

        # define location prediction as sum of top three beer predictions
        if len(l['beers'])>=3:
            l['prediction'] = np.sum( [i['item']['prediction'] for i in l['beers']][0:3] )
        else: # or all beer predictions if less than three
            l['prediction'] = np.sum( [i['item']['prediction'] for i in l['beers']] )

    # sort beers and locations descending by prediction
    user_beers.sort( key=lambda x:x['prediction'],reverse=True )
    user_loc.sort( key=lambda x:x['prediction'],reverse=True )

    d['beers'] = user_beers
    d['locations'] = user_loc


def saveUserData(user,d):
    # first load existing from disk
    fnam = 'userdata/'+user+'.pklz2'
    import gzip,cPickle,json
    with gzip.open(dat_dir+fnam,'rb') as f: 
        udat = cPickle.load(f)

    # add predictions
    udat['beers'] = d['beers']
    udat['locations'] = d['locations']

    # save
    with gzip.open(dat_dir+fnam,'wb') as f: 
        cPickle.dump(udat,f,protocol=2)


def run(user):
    show = False

    d = loadData()
    loadUserData(user,d,show=show)
    train(d)
    predict(d,show=show)
    findBestLocations(d)
    saveUserData(user,d)
    
    return (d['beers'],d['locations'])


