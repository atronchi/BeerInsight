# working from tutorial at: http://docs.cython.org/src/tutorial/numpy.html
# to compile: python setup_sparse_mult.py build_ext --inplace
cimport numpy as np

# Turn bounds checking back on if there are ANY problems!
cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def sparse_mult_c(np.ndarray[np.float64_t, ndim=2] a, 
                  np.ndarray[np.float64_t, ndim=2] b, 
                  np.ndarray[np.int32_t, ndim=1] rows, 
                  np.ndarray[np.int32_t, ndim=1] cols,
                  np.ndarray[np.float64_t, ndim=1] C ):

    cdef int n = rows.shape[0]
    cdef int k = a.shape[1]
    cdef int i,j
    
    for i in range(n):
        for j in range(k):
            C[i] += a[rows[i],j] * b[j,cols[i]]


# cython-ized version of sparse_mult: works fast, actually sparse
import numpy as np
import scipy.sparse as sp
def sparse_mult(a, b, coords):
    # inspired by handy snippet from 
    # http://stackoverflow.com/questions/13731405/calculate-subset-of-matrix-multiplication
    #a,b are np.ndarrays
    rows, cols = coords
    C = np.zeros(rows.shape[0])
    sparse_mult_c(a,b,rows,cols,C)
    return sp.coo_matrix( (C,coords), (a.shape[0],b.shape[1]) )


