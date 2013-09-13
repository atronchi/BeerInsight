import gzip,cPickle
import numpy as np

def loadData(fnam):
    with gzip.open(fnam,'rb') as f: return cPickle.load(f)

BA = loadData('recommender/reviews.pklz2')
RB = loadData('scraper/scrape_ratebeer.pklz')

BA_beers = BA['beers']
RB_beers = np.array([ [i['brewery'],i['name']] for i in RB['beers'] ])

# generate index mapping each RB entry to the appropriate BA entry
RB_to_BA = np.zeros( len(RB_beers), dtype=int )-1

#for i in enumerate(RB_beers):
    
