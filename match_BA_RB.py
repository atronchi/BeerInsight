import gzip,cPickle

def loadData(fnam):
    with gzip.open(fnam,'rb') as f: return cPickle.load(f)

BA = loadData('recommender/reviews.pklz2')
RB = loadData('scraper/scrape_ratebeer.pklz')

# generate index mapping each RB entry to the appropriate BA entry
#RB_to_BA = 

