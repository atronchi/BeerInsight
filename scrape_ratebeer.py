#!/usr/bin/python
# Scrape ratebeer website for beer location data

import requests
import re
import BeautifulSoup
import numpy as np

# functions to scrape from rb
def parse_area(url):
    global domain
    domain = url.split('/')[2]
    r = requests.get(url)
    con = unicode(r.content,r.encoding).split('\r\n')

    locations=[]
    for c in con:
        # dataTableRow class represents beer locations
        if 'dataTableRow' in c:
            # parse the location line
            loc_url = 'http://'+domain+ c.split('<a href="')[1].split('">')[0]
            loc_name = c.split('<strong>')[1].split('<')[0].strip()
            loc_id = c.split('print-')[1].split('.htm')[0] # use this id in the ajax scripts to return beers here?
            loc_type = c.split('<span class="bold_')[1].split('">')[1].split('<')[0]
            loc_tel = c.split('<a href="tel:')[1].split('">')[0]
            loc_score = c.split('<span class=lineNumber>')[1].split('<')[0]
            loc_addr = c.split('border=0>')[1].split('<')[0].strip()

            print ' parsing shop: '+loc_name, 
            (loc_addr2,beers) = parse_loc(loc_url,loc_id)
            if loc_addr2 != u'': loc_addr = loc_addr2
            print ', {0} beers here'.format(len(beers))
            if loc_name == '': loc_name=loc_addr

            # do geocoding
            import time
            time.sleep(1) # 1s sleep to prevent geocode from rate-limiting
            # https://developers.google.com/maps/documentation/business/faq?csw=1#usage_limits
            gc = geocode(loc_addr)
            loc_latlng = gc['latlng']
            loc_addr = gc['addr_formatted']

            # step into url to retrieve more info
            locations.append({
                'name':loc_name,
                'id':loc_id,
                'type':loc_type,
                'tel':loc_tel,
                'rb_score':loc_score,
                'addr':loc_addr,
                'latlng':loc_latlng,
                'url':loc_url,
                'beers':beers
                })

    return locations


def parse_loc(loc_url,loc_id):
    loc = requests.get(loc_url)
    loc_con = unicode(loc.content,loc.encoding).split('\r\n')
    beers = []
    loc_addr = u''
    for l in loc_con:
        if '<br>Type:' in l:
            loc_addr = l.split('<br>Type:')[1].split('<br><br>')[1].split('<a href')[0].replace('<br>',', ').strip()
        if '<a title=' in l:
            beers += parse_beerln(l)
        if 'More Beers Available Here' in l:
            mb = requests.get('http://www.ratebeer.com/plaCes/showplacebeers.asp?pid='+loc_id)
            mc = unicode(mb.content,mb.encoding)
            beers += parse_beerln(mc)
    return (loc_addr,beers)
    

def parse_beerln(ln):
    global domain
    beers =  []
    ln = ln.split('<a title=')[1:]
    for a in ln:
        a=a.split('"')
        beers.append({
            'date' : a[1],
            'url' : 'http://'+domain+a[3],
            'name' : a[4].split('>')[1].split('<')[0],
            'rb_score' : a[4].split('>')[3].split('<')[0]
            })
    return beers


# scrape deeper beer info from beers found at locations
def parse_beer_deep(url):
    loc = requests.get(url)
    global domain
    domain = url.split('/')[2]
    
    '''
    beautiful soup looks like a cool way to parse HTML, but I can't figure out the parser for some tags so I'm going with what I know...
    soup = BeautifulSoup.BeautifulSoup(loc.content)
    name = soup.h1.text
    '''

    l = unicode(loc.content,loc.encoding)

    if 'Proceed to the aliased beer...' in l:
        alias_url = 'http://'+domain+\
            l.split('Proceed to the aliased beer...')[1].split('<A HREF="')[1].split('"')[0]
        return parse_beer_deep(alias_url)

    if 'Is Unlisted' in l:
        return {}

    if '<h1>' in l:
        name = l.split('<h1>')[1].split('</h1>')[0]
        brewery = l.split('/brewers/')[1].split('>')[1].split('<')[0]
        #brewery = l.split('Brewed at')[1].split('>')[1].split('<')[0] # sometimes 'brewed at' sometimes 'brewed in' ...
        style = l.split('Style:')[1].split('>')[1].split('<')[0]
        if not 'No Score' in l:
            rb_score = l.split('overall')[1].split('<br>')[1]
        else:
            rb_score = -1

    return {'name':name
           ,'brewery':brewery
           ,'style':style
           ,'rb_score':rb_score
           ,'url':url
           }
            

def saveData(result,fnam=''):
    # pickle for later
    import cPickle,gzip,sys,time # see also shelve and shove

    if fnam=='':fnam = 'scrape_ratebeer '+time.ctime(time.time())+'.pklz'

    print 'saving '+fnam+' ...',; sys.stdout.flush()

    with gzip.open(fnam,'wb') as f:
        cPickle.dump(result, f)

    print 'done.'


def print_unique():
    # figure out how many unique beers we know about in the area
    all_beers=[]
    for l in locations: all_beers += l['beers']
    set_beers = set([ i['item']['name'] for i in all_beers ])
    print 'there are {0} unique beers in the area'.format(len(set_beers))


# query google to geocode addresses
def geocode(addr = '1600 Amphitheatre Parkway, Mountain View, CA 94043, USA'):
    import json,requests

    # as per: https://developers.google.com/maps/documentation/geocoding/#GeocodingRequests
    r=requests.get('http://maps.googleapis.com/maps/api/geocode/json' + 
                   '?address={0}&sensor=true'.format(addr.replace(' ','+'))
                   )

    j=json.loads(unicode(r.content,r.encoding))
    stat = j['status']
    if stat=='OK':
        addr_formatted = j['results'][0]['formatted_address']
        g = j['results'][0]['geometry']['location']
        latlng = [g['lat'],g['lng']]
    else:
        addr_formatted = addr
        latlng = [0,0]
    print ' geocode '+stat+', latlng={0} for address: '.format(latlng)+addr


    #return j
    return {'status':stat,'addr_formatted':addr_formatted,'latlng':latlng}


# Make it so!
url = 'http://www.ratebeer.com/places/regions/san-jose/7400/5/'
print 'parsing area url: '+url
locations = parse_area(url)


# build set of beer urls
beer_urls = []
for l in locations:
    for b in l['beers']:
            beer_urls.append(b['url'])
beer_urls=np.array(list(set(beer_urls)))

# deep beer scraping
beers = []
for u in beer_urls:
    print 'parsing beer url: '+u
    pbd = parse_beer_deep(u)
    beers.append( pbd )

# update locations with new scraped data
for l in enumerate(locations):
    del_idxs=[]
    for b in enumerate(l[1]['beers']):
        idx = np.where( beer_urls == b[1]['url'] )[0][0]
        if beers[idx] != {}:
            locations[l[0]]['beers'][b[0]] = {
               'date':b[1]['date']
              ,'item':beers[idx]
              }
        else: # empty dict returned for unlisted beers
            del_idxs.append(b[0])
    if del_idxs != []: # delete any unlisted beers
        locations[l[0]]['beers'] = [i for j, i in enumerate(l[1]['beers']) if j not in del_idxs]

result = {'area_url':url,'locations':locations}
saveData(result)


