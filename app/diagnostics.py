#  From the matrix, we can compute statistics like average rating.
def avgRating(i=0):
    r = np.array( R[:,i].todense() ).flatten()
    m = movieList[i]
    whr = np.where(r==1)[0]
    if np.prod(whr.shape) > 0:
        mean = Y[whr,i].mean()
    else:
        mean = '*' 
    print 'Average rating for '.format(i) + m + ': {0} / 5\n'.format(mean)


# We can "visualize" the ratings matrix by plotting it with imagesc
def plotRatings():
    ''' imshow and matshow don't work for sparse matrices
    # see http://stackoverflow.com/questions/18748929/matshow-with-sparse-matrices
    '''
    pl.figure()
    pl.spy(Y,marker='.',markersize=1)
    pl.ylabel('users')
    pl.xlabel('movies')

    users_nrev = np.array( R.sum(axis=0) ).flatten()
    movies_nrev = np.array( R.sum(axis=1) ).flatten()
    movies_sumrev = np.array( Y.sum(axis=1) ).flatten() 
    movies_meanrev = movies_sumrev / movies_nrev

    pl.figure()
    pl.hist(users_nrev, bins=50, log=True)
    pl.ylabel('number of users')
    pl.xlabel('number of reviews')
    pl.title('user review distribution')

    pl.figure()
    pl.hist(movies_nrev, bins=50, log=True)
    pl.ylabel('number of movies')
    pl.xlabel('number of reviews')
    pl.title('movie review distribution')

    pl.figure()
    pl.hist(movies_meanrev, bins=50)
    pl.ylabel('number of movies')
    pl.xlabel('rating')
    pl.title('movie rating distribution')

    return (users_nrev,movies_nrev,movies_sumrev,movies_meanrev)

#(users_nrev,movies_nrev,movies_sumrev,movies_meanrev) = plotRatings()




