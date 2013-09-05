#!/usr/bin/python
# Scrape ratebeer website for beer location data

import requests
import re

url = 'http://www.ratebeer.com/places/regions/san-jose/7400/5/'
domain = url.split('/')[2]

r = requests.get(url)
con = r.content.split('\r\n')

# parse for locations
places = []
for c in con:
    # dataTableRow class represents beer locations
    if 'dataTableRow' in c:
        # parse the location line
        loc_url = c.split('<a href="')[1].split('">')[0]
        loc_name = c.split('<strong>')[1].split('<')[0]
        loc_id = c.split('print-')[1].split('.htm')[0] # use this id in the ajax scripts to return beers here?
        loc_type = c.split('<span class="bold_brown">')[1].split('<')[0]
        loc_tel = c.split('<a href="tel:')[1].split('">')[0]
        loc_score = c.split('<span class=lineNumber>')[1].split('<')[0]

        # step into url to retrieve more info
        loc = requests.get(loc_url)
        loc_con = r.content.split('\r\n')
        loc_beers = []
        for l in loc_con:
            if '<a title=' in l:
                parse_beer(l)
            if 'More Beers Available Here' in l:
                more_beers = requests.get('http://www.ratebeer.com/plaCes/showplacebeers.asp?pid='+loc_id)
                parse_beer(more_beers.content)

'''
        places.append(
            {'
'''

