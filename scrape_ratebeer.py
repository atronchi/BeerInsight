#!/usr/bin/python
# Scrape ratebeer website for beer location data

import requests
import re

# functions to scrape from rb
def parse_beer(ln):
    beers =  []
    ln = ln.split('<a title=')[1:]
    for a in ln:
        a=a.split('"')
        beers.append({
            'date' : a[1],
            'url' : a[3],
            'name' : a[4].split('>')[1].split('<')[0],
            'rb_score' : a[4].split('>')[3].split('<')[0]
            })
    return beers


def parse_loc(loc_url,loc_id):
    loc = requests.get(loc_url)
    loc_con = unicode(loc.content,loc.encoding).split('\r\n')
    beers = []
    loc_addr = u''
    for l in loc_con:
        if '<br>Type:' in l:
            loc_addr = l.split('<br>Type:')[1].split('<br><br>')[1].split('<a href')[0].replace('<br>',', ').strip()
        if '<a title=' in l:
            beers += parse_beer(l)
        if 'More Beers Available Here' in l:
            mb = requests.get('http://www.ratebeer.com/plaCes/showplacebeers.asp?pid='+loc_id)
            mc = unicode(mb.content,mb.encoding)
            beers += parse_beer(mc)
    return (loc_addr,beers)
    

def parse_area(url):
    domain = url.split('/')[2]
    r = requests.get(url)
    con = unicode(r.content,r.encoding).split('\r\n')

    locations=[]
    for c in con:
        # dataTableRow class represents beer locations
        if 'dataTableRow' in c:
            # parse the location line
            loc_url = 'http://'+domain+ c.split('<a href="')[1].split('">')[0]
            loc_name = c.split('<strong>')[1].split('<')[0]
            loc_id = c.split('print-')[1].split('.htm')[0] # use this id in the ajax scripts to return beers here?
            loc_type = c.split('<span class="bold_')[1].split('">')[1].split('<')[0]
            loc_tel = c.split('<a href="tel:')[1].split('">')[0]
            loc_score = c.split('<span class=lineNumber>')[1].split('<')[0]
            loc_addr = c.split('border=0>')[1].split('<')[0].strip()

            (loc_addr2,beers) = parse_loc(loc_url,loc_id)
            if loc_addr2 != u'': loc_addr = loc_addr2

            # step into url to retrieve more info
            locations.append({
                'name':loc_name,
                'id':loc_id,
                'type':loc_type,
                'tel':loc_tel,
                'rb_score':loc_score,
                'addr':loc_addr,
                'beers':beers
                })

    return locations


# Make it so!
url = 'http://www.ratebeer.com/places/regions/san-jose/7400/5/'
locations = parse_area(url)


# pickle for later
import pickle,gzip,sys,time # see also shelve and shove
rb_file = 'scrape_ratebeer '+time.ctime(time.time())+'.pklz'
print 'saving '+rb_file+' ...',; sys.stdout.flush()
with gzip.open(rb_file,'wb') as f:
    pickle.dump({'area_url':url,'locations':locations}, f)
print 'done.'


# figure out how many unique beers we know about in the area
all_beers=[]
for l in locations: all_beers += l['beers']
set_beers = set([ i['name'] for i in all_beers ])
print 'there are {0} unique beers in the area'.format(len(set_beers))

