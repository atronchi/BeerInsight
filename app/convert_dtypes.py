import gzip,cPickle,pickle
import scipy.sparse as sp

def loadData(fnam):
    with gzip.open(fnam,'rb') as f: return cPickle.load(f)

def saveData(obj,fnam):
    with gzip.open(fnam,'wb') as f: cPickle.dump(obj,f,protocol=2)

dat = loadData('review_matrix.pklz')

dat['ratings'] = sp.csr_matrix( dat['ratings']*2, dtype='uint8' )
dat['israted'] = sp.csr_matrix( dat['israted'], dtype='bool' )

saveData(dat,'reviews.pklz2')

