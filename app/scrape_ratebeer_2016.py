#!/usr/bin/python
# Scrape ratebeer website for beer location data

import BeautifulSoup as bs
import numpy as np
import re
import requests

# e.g. seed = get_seed()
def get_seed():
    # use this to seed brewers, beers, and users
    # then load their specific pages for detailed info

    brewers = set()
    beers = set()
    users = set()
    reviews = dict()

    rating_url = 'http://www.ratebeer.com/beer-ratings/'
    # page loads a <table ... class="table"> with several reviews
    # <tr>
    #  (a brewer) <td><a href="{brewer}">...
    #  <td><a href="{beer}"> ...<div><a href="{user}"
    # </tr> and maybe 20 more or so

    # after the first 100 pages, the 100th page is just repeated so don't bother seeding more
    for url in [rating_url + '0/{}/'.format(n) for n in range(1, 101)]:
        print('scraping {} ...'.format(url))
        soup = bs.BeautifulSoup(requests.get(url).content)
        for tr in soup.table('tr'):
            td = tr('td')
            brewer_match = re.search('<a href="/brewers/x/(.+?)/">', str(td[0]))
            if brewer_match:
                br_id = int(brewer_match.group(1))
                brewers.add(br_id)

            beer_match = re.search('<a .+? href="/beer/(.+?)/">(.+?)</a>', str(td[1]))
            if beer_match:
                (b_name, b_id) = beer_match.group(1).split('/')
                b_longname = beer_match.group(2)
                beers.add((int(b_id), b_name, b_longname))

            user_match = re.search(
                '<a href="/user/.+?"></a><a href="/user/(.+?)/">(.+?)</a>', str(td[1])
            )
            if user_match:
                (u_id, u_name) = user_match.groups()
                users.add((int(u_id), u_name))

            if beer_match and user_match:
                rating_match = re.search('<span class="uas">(.+?)</span>', str(td[1]))
                review_match = re.search('<span style=".+?">(.+?)</span>', str(td[1]))
                rating = rating_match.group(1) if rating_match else None
                review = review_match.group(1) if review_match else None
                reviews[int(u_id), int(b_id)] = (float(rating), review)

    print('scraped {0} brewers, {1} beers, {2} users, and {3} reviews.').format(
        len(brewers), len(beers), len(users), len(reviews)
    )

    return {
        # brewer urls are http://www.ratebeer.com/brewers/x/{br_id}/
        'brewer': brewers,
        # beer urls are http://www.ratebeer.com/beer/{b_name}/{b_id}/
        'beer': beers,
        # user urls are http://www.ratebeer.com/user/{u_id}/
        'user': users,
        'review': reviews
    }

def get_beer(url):
    # url format is http://www.ratebeer.com/beer/flying-dog-bloodline-blood-orange-ipa/209983/
    # or http://www.ratebeer.com/beer/x/209983/  (first slash is irrelevant?)
    beer_url = 'http://www.ratebeer.com/beer/{0}/'.format(url)
    # page loads metadata for the beer
    # <span 

    pass

def get_user(url):
    # url format is http://www.ratebeer.com/user/{u_id}/
    pass


# reviews
# breweries

def get_places():
    # http://www.ratebeer.com/places/browse/
    pass

