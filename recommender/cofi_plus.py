#!/usr/bin/python
# -*- coding: utf8 -*-
# This is a modified python version of the exercise for
# Coursera ml-003 machine learning taught by Andrew Ng
# written by Alexander Tronchin-James 2013-05-31
# original path /home/alex/Google Drive/coursera/ml-003/ex8
import os
import numpy as np
import pylab as pl
from matplotlib import rc; rc('text', usetex=True)
from scipy.io import loadmat
from scipy.optimize import minimize
import ipdb # not use ipdb.set_trace()
from timing import *


## Initialization
os.system('clear') # clear the screen
pl.close('all') # close any open plots
pl.ion() # turn on plotting interactive mode, otherwise plots pause computation


## =============== Part 1: Loading ratings dataset ================
def loadData(fnam='result_clean.pklz'):
    # load the processed data

    import cPickle,gzip,sys
    print 'loading '+fnam+' ...',; sys.stdout.flush()

    toc = tic()
    with gzip.open(fnam,'rb') as f: result = cPickle.load(f)
    
    print 'done in {0}s.'.format(tic()-toc)

    return result


result = loadData() # this can take 15min so only load if not already in memory

Y = result['ratings']
R = result['israted']
users = result['users']
beers = result['beers']
user_nrev = result['user_nrev']
beer_nrev = result['beer_nrev']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating for a beer
def avgRating(i=0):
    r = np.array( R[:,i].todense() ).flatten()
    b = beers[i][0]+', '+beers[i][1]
    whr = np.where(r==1)[0]
    if np.prod(whr.shape) > 0:
        mean = Y[whr,i].mean()
    else:
        mean = '*'
    print 'Average rating for '.format(i) + b + ': {0} / 5\n'.format(mean)

avgRating(0)


'''
## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
matdat2=loadmat('ex8_movieParams.mat')
X = np.matrix(matdat2['X'])
Theta = np.matrix(matdat2['Theta'])


#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
shapes = (num_users,num_movies,num_features)

X = X[:num_movies,:num_features]
Theta = Theta[:num_users,:num_features]
Y = Y[:num_movies,:num_users]
R = R[:num_movies,:num_users]

# Define cost function
def cofiCostFunc(params, Y,R, shapes, Lambda, debug=False):
    (num_users,num_movies,num_features)=shapes

    # reshape params into X and Theta
    X = np.matrix(np.reshape(
        params[:num_movies*num_features], 
        (num_movies, num_features), order='F'
        ))
    Theta = np.matrix(np.reshape(
        params[num_movies*num_features:],
        (num_users, num_features), order='F'
        ))
    Lambda=float(Lambda) # ensure Lambda isn't an int!
    
    XThmY = X*Theta.T - Y
    XThmYR= np.matrix( np.array(XThmY)*np.array(R) )

    J = 1/2. *( np.array(XThmYR)**2 ).sum()
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

#  Evaluate cost function
test_theta = np.hstack((
    np.array(X).flatten('F'), 
    np.array(Theta).flatten('F') 
    ))
(J,grad) = cofiCostFunc(test_theta, Y, R, shapes, 0)

print 'Cost at loaded parameters: {0}'.format(J) + \
      '\n(this value should be about 22.22)\n'

#print 'Program paused. Press enter to continue.'; raw_input()


## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
print '\nChecking Gradients (without regularization) ... '

#  Check gradients by running checkNNGradients
def computeNumericalGradient(theta,Y,R,shapes,Lambda):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1.e-4
    for p in range(len(theta)):
        # set perturbation vector
        perturb[p] = e
        (loss1, dum) = cofiCostFunc(theta-perturb,Y,R,shapes,Lambda)
        (loss2, dum) = cofiCostFunc(theta+perturb,Y,R,shapes,Lambda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def checkCostFunction(Lambda=0):
    # create small problem
    X_t = np.matrix(np.random.rand(4,3))
    Theta_t = np.matrix(np.random.rand(5,3))

    # zap out most entries
    Y = X_t*Theta_t.T
    rands = np.random.rand(Y.shape[0],Y.shape[1])
    Y[ rands>0.5 ] = 0
    R = np.zeros(Y.shape)
    R[ Y!=0 ] = 1

    X = np.random.randn(X_t.shape[0],X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0],Theta_t.shape[1])

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    shapes = (num_users,num_movies,num_features)
    
    grad_t = np.hstack((
        np.array(X).flatten('F'), 
        np.array(Theta).flatten('F') 
        ))
    numgrad = computeNumericalGradient( grad_t,Y,R,shapes,Lambda )
    (cost,grad) = cofiCostFunc( grad_t,Y,R,shapes,Lambda )
    
    print np.vstack((numgrad,grad)).T
    print 'The above two columns you get should be very similar.\n' + \
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n'
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad);
    print 'If your backpropagation implementation is correct, then \n' + \
          'the relative difference will be small (less than 1e-9). \n' + \
          '\nRelative Difference: {0}\n'.format(diff)

checkCostFunction()

#print 'Program paused. Press enter to continue.'; raw_input()


## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  

#  Evaluate cost function
test_theta = np.hstack((
    np.array(X).flatten('F'), 
    np.array(Theta).flatten('F') 
    ))
(J,grad) = cofiCostFunc( test_theta,Y,R,shapes,1.5 )
           
print 'Cost at loaded parameters (lambda = 1.5): {0}'.format(J) + \
      '\n(this value should be about 31.34)'

#print 'Program paused. Press enter to continue.'; raw_input()


## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#

#  Check gradients by running checkNNGradients
print '\nChecking Gradients (with regularization) ... '
checkCostFunction(1.5)

#print 'Program paused. Press enter to continue.'; raw_input()


## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#

def loadMovieList():
    with open('movie_ids.txt','r') as f:
        movieList = f.readlines()
    return np.array([' '.join(l.split()[1:]) for l in movieList])

movieList = loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros((len(movieList),1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1] = 4

# Or suppose did not enjoy Silence of the Lambs [1991], you can set
my_ratings[98] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[7] = 3
my_ratings[12]= 5
my_ratings[54] = 4
my_ratings[64]= 5
my_ratings[66]= 3
my_ratings[69] = 5
my_ratings[183] = 4
my_ratings[226] = 5
my_ratings[355]= 5

# remove first element of my_ratings and append a new zero so indexing agrees with python
my_ratings = np.vstack(( my_ratings[1:],np.zeros((1,1)) ))

print 'New user ratings:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print 'Rated {0} for {1}'.format(my_ratings[i,0],movieList[i])

#print 'Program paused. Press enter to continue.'; raw_input()


## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#

#  Re-load data
matdat1=loadmat('ex8_movies.mat')
Y = np.matrix(matdat1['Y'])
R = np.matrix(matdat1['R'])
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10 # adjustable
shapes = (num_users,num_movies,num_features)

#  Normalize Ratings
def normalizeRatings(Y,R):
    (m,n) = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where( R[i,:] == 1 )[1]
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    return (Ynorm,Ymean)

print '\nNormalizing ratings...'
[Ynorm, Ymean] = normalizeRatings(Y, R);


# Set Initial Parameters (Theta, X), regularization, and costFn
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.hstack((
    np.array(X).flatten('F'), 
    np.array(Theta).flatten('F') 
    ))
Lambda = 10
costFn = lambda t: cofiCostFunc( t,Ynorm,R,shapes,Lambda )

# Train
print '\nTraining collaborative filtering...\n'
result = minimize(costFn, initial_parameters,
                  method='L-BFGS-B', jac=True, #callback=cb, 
                  options={'disp':True,'maxiter':100}
                 ) 
(cost,params) = (result['fun'],result['x'])

# Unfold the returned theta back into U and W
X = np.matrix(np.reshape(
    params[:num_movies*num_features], 
    (num_movies, num_features), order='F'
    ))
Theta = np.matrix(np.reshape(
    params[num_movies*num_features:],
    (num_users, num_features), order='F'
    ))

print 'Recommender system learning completed.\n'

#print 'Program paused. Press enter to continue.'; raw_input()


## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = X*Theta.T
my_predictions = np.array(p[:,0]).flatten() + Ymean

movieList = loadMovieList()

#[r, ix] = sort(my_predictions, 'descend');
ix = [i[0] for i in sorted(
    enumerate(my_predictions), 
    key=lambda x:x[1], 
    reverse=True
    )]

print '\nTop recommendations for you:'
for i in range(10):
    j = ix[i]
    print 'Predicting rating {0} for movie {1}'.format(
        my_predictions[j],
        movieList[j]
        )

print '\nOriginal ratings provided:'
for i in range(len(my_ratings)):
    if my_ratings[i] > 0: 
        print 'Rated {0} for {1}'.format(my_ratings[i][0],movieList[i])


# split into test and validation sets
# tune regularization parameter and number of features
'''
