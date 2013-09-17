import gzip,cPickle
import numpy as np

with gzip.open('recommender/reviews.pklz2') as f: BA=cPickle.load(f)

# select 1000 most frequently reviewed beers for prefetch
brewer_prefetch = list(set( 
    [b[0] for b in BA['beers'][ np.argsort(BA['beer_nrev'])[0:1000] ]] 
    ))
brewers_n_beers={}
for b in BA['beers']:
    if b[0]!='':
        if b[0] in brewers_n_beers.keys():
            brewers_n_beers[b[0]].append(b[1])
        else:
            brewers_n_beers[b[0]]=[b[1]]

tpf = {
    'brewer_prefetch':brewer_prefetch,
    'brewers_n_beers':brewers_n_beers
    }

with gzip.open('typeahead_prefetch.pklz2','wb') as f: cPickle.dump(tpf,f,protocol=2)

#with gzip.open('typeahead_prefetch.pklz2','rb') as f: tpf_load=cPickle.load(f)

